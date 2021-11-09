#include "men.h"
#include <ctime>
#include <vector>
#include "arithmatic_coder.h"
#include <stdlib.h>

using namespace std;

using namespace torch;


void fill_pros(int32_t *rel_out){
    for(uint i=0;i<num_max_batch_size;i++){
        probabilities[i][0].c=characters[0];
        probabilities[i][0].low=0;
        probabilities[i][0].high=rel_out[i*num_ch+0];
        probabilities[i][0].scale=rel_out[i*num_ch+num_ch-1];
        for(uint j=1;j<num_ch;j++){
            probabilities[i][j].c=characters[j];
            probabilities[i][j].low=rel_out[i*num_ch+j-1];
            probabilities[i][j].high=rel_out[i*num_ch+j];
            probabilities[i][j].scale=rel_out[i*num_ch+num_ch-1];

        }
    }
}



int main(int argc, const char* argv[]) {
    if(argc<8){
        cout<<"usage: DeepQC <c|x> params_file model.pt input_file qv_file output_file alpha\n";
        cout<<"\n<c|x>: c for compress; x for decompress\n";
        cout<<"alpha: indicates how many reads concatenated together in a row of a parallel group. \n    we suggest set alpha to 20 when the quality score file size smaller than 100MB, 1000 when larger than 1GB.\n";
        return 0;
    }

    config_params(argv[2], atoi(argv[7]));
    BitIO *ios=new BitIO[num_max_batch_size];
    Coder *coders=new Coder[num_max_batch_size];
    mem *mems=new mem[num_max_batch_size];

    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    torch::DeviceType cpu_type;
    cpu_type = torch::kCPU;
    torch::Device cpu(cpu_type);
    auto module = torch::jit::load(argv[3]);
    module.to(device);

    if(argv[1][0]=='c'){
        ofstream outFile(argv[6],ios::binary);
        input a(argv[4],argv[5]);

        if(a.inputFile.good() && a.inputFile_qv.good()){
            a.refresh();
            
            if(num_batch){
                for(uint64_t i=0;i<num_batch;i++) {
                    mems[i].start = 0;
                    ios[i].initialize_output_bitstream();
                    coders[i].initialize_arithmetic_encoder(i);
                }
                cout<<"compress 111"<<endl;
                uint64_t f = 0;
                while(f<64) {
                    auto data=a.get_col(f);
                    for(uint64_t i=0;i<num_batch;i++) {
                        if((i*num_reads_per_batch+f)<len_reads)
                            coders[i].encode_symbol(mems[i], &st_pro[data[i]], ios[i], i);
                    }
                    f++;
                }
                cout<<"compress 222"<<endl;
                while(f<num_reads_per_batch){
                    auto data=a.get_col(f);
                    auto prebase=a.get_prebase();
                    auto preqv=a.get_preqv();
                    auto output = module.forward({prebase.to(torch::kLong).to(device), preqv.to(torch::kLong).to(device)}).toTensor().to(cpu).detach();
                    output=output.contiguous();

                    auto rel_out=output.data_ptr<int32_t>();
                    fill_pros(rel_out);
                    for(uint64_t i=0;i<num_batch;i++){
                        if((i*num_reads_per_batch+f)<len_reads)
                            coders[i].encode_symbol(mems[i],&probabilities[i][data[i]],ios[i],i);
                    }
                    f++;
                }

                cout<<"compress 333"<<endl;
                vector<uint32_t> out_size(num_batch);
                for (int k = 0; k <num_batch ; ++k) {
                    coders[k].flush_arithmetic_encoder(mems[k],ios[k],k);
                    ios[k].flush_output_bitstream(mems[k]);
                    out_size[k]=mems[k].true_size;
                }

                memToFile mem(num_batch,out_size);
                mem.writeToMem(mems,out_size);
                mem.writeToFile(outFile);
                outFile.flush();
            }
        }
        outFile.close();
        cout<<"compress finish"<<endl;
    }
    
    else if(argv[1][0]=='x'){
        fileToMem fm(argv[4]);
        ofstream outfile(argv[6]);
        cout<<"decompress 111"<<endl;
        if (fm.refresh()){
            tempDecoder td(argv[5]);
            cout<<"decompress 222"<<endl;
            for(uint64_t i=0;i<num_batch;i++) {
                fm.mems[i].start=0;
                ios[i].initialize_input_bitstream();
                coders[i].initialize_arithmetic_decoder(fm.mems[i] ,ios[i]);
            }
            uint32_t count;
            uint64_t j=0;
            cout<<"decompress 333"<<endl;
            for (;j<64;j++){
                for(uint64_t i=0;i<num_batch;i++) {
                    if((i*num_reads_per_batch+j)<len_reads){
                        count=coders[i].get_current_count(&st_pro[num_ch-1]);
                        for (int k = num_ch-1; k >=0; --k) {
                            if (count >= st_pro[k].low && count < st_pro[k].high) {
                                coders[i].remove_symbol_from_stream(fm.mems[i],&st_pro[k],ios[i]);
                                td.outputs[i][j]=characters[k];
                                break;
                            }
                        }
                    }
                }
            }
            cout<<"decompress 444"<<endl;
            for (;j<num_reads_per_batch;j++){
                auto prebase=td.get_prebase();
                auto preqv=td.get_preqv();
                auto output = module.forward({prebase.to(torch::kLong).to(device), preqv.to(torch::kLong).to(device)}).toTensor().to(cpu).detach();
                output=output.contiguous();
                auto rel_out=output.data_ptr<int32_t >();
                fill_pros(rel_out);
                for(uint64_t i=0;i<num_batch;i++){
                    if((i*num_reads_per_batch+j)<len_reads){
                        count=coders[i].get_current_count(&probabilities[i][num_ch-1]);
                        for (int k = num_ch-1; k >=0; --k) {
                            if (count >= probabilities[i][k].low && count < probabilities[i][k].high) {
                                coders[i].remove_symbol_from_stream(fm.mems[i],&probabilities[i][k],ios[i]);
                                td.outputs[i][j]=characters[k];
                                break;
                            }
                        }
                    }
                }
            }
            cout<<"decompress 555"<<endl;
            td.writeQs(outfile);
        }
        outfile.close();
        cout<<"decompress finish\n";
    }

    delete []coders;
    delete []ios;
    delete []mems;

    return 0;
}

