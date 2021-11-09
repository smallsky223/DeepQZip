#pragma once
//
// Created by qiusuo on 2019/6/2.
//

#ifndef LSTM_COMPRESSION_MEM_H
#define LSTM_COMPRESSION_MEM_H

#include <cstdint>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "arithmatic_coder.h"

//num_max_seq_len 最大的序列长读
//num_reads_per_batch 每个批次有多少行序列
//num_max_batch_size 最大的批次数量

using namespace std;


SYMBOL probabilities[num_max_batch_size][8]{
        {
                { 'B',  0,  1  },
                { 'I',  1,  2  },
                { 'L',  2,  4  }
        }
};
uint64_t num_batch = 100;
extern uint64_t num_batch;
//const uint64_t binary_map_size = 536870912;//512M

//const uint64_t one_out_size=536870912*2/num_max_batch_size;

SYMBOL st_pro[8];
char characters[8];
uint32_t num_ch=4;
uint64_t len_reads = 10000;
uint64_t num_reads_per_batch = 100;

char char2int[128];
//__uint32_t statisticCum[35];

void config_params(const char* stat_info, int alpha){
    ifstream st(stat_info);
    st>>num_ch;
    for (int l = 0; l < num_ch; ++l) {
        st>>characters[l];
    }
    st>>len_reads;
    num_reads_per_batch = len_reads / alpha;
    if(len_reads%num_reads_per_batch)
        num_batch = len_reads / num_reads_per_batch + 1;
    else
        num_batch = len_reads / num_reads_per_batch;
    

    for(uint i=0;i<num_ch;i++){
//			statisticCum[i]=i+1;
        char2int[characters[i]]=i;
    }

    for(int i=0;i<num_ch;i++){
        st_pro[i].c = characters[i];
        st_pro[i].low = i*10;
        st_pro[i].high = (i+1)*10;
        st_pro[i].scale = num_ch*10;
    }

    st.close();

}
class input {
public:
	char *content_b;
    char *content_q;
	ifstream inputFile;
    ifstream inputFile_qv;
	at::Tensor prebase=torch::zeros({num_max_batch_size, 64},torch::TensorOptions().dtype(torch::kInt8));
    at::Tensor preqv=torch::zeros({num_max_batch_size, 64},torch::TensorOptions().dtype(torch::kInt8));
    char one_col[num_max_batch_size];
	char one_precol[num_max_batch_size][64];
    uint64_t precol_index = 0;
	input(string fileName, string fileName_qv) {
		content_b = (char*)malloc(num_batch * num_reads_per_batch);
        content_q = (char*)malloc(num_batch * num_reads_per_batch);
		inputFile.open(fileName);
        inputFile_qv.open(fileName_qv);
	}
	void refresh() {
        char buf[1];
        char buf_qv[1];
        for (uint64_t i = 0; i < len_reads; i++){
            inputFile.read(buf,1);
            content_b[i] = char2int[buf[0]];
        }
        for (uint64_t i = 0; i < len_reads; i++){
            inputFile_qv.read(buf_qv,1);
            content_q[i] = buf_qv[0]-' ';
        }
        precol_index = 0;
	}
    
    // 获取每个batch的index上的数据，即待编码字符
	char* get_col(uint64_t col_index) {
		for (uint64_t i = 0; i < num_batch; i++) {
			one_col[i]=content_b[i*num_reads_per_batch + col_index];
		}
		return one_col;
	}
    
    // 获取每个batch上的待编码字符的base上下文
	at::Tensor& get_prebase() {
		for (uint64_t i = 0; i < num_batch; i++) {
            for (uint64_t j = 0; j < 64; j++) {
                one_precol[i][j]=content_b[i*num_reads_per_batch + precol_index + j];
            }
		}
        for (uint64_t i = 0; i < num_batch; i++){
            memcpy(prebase[i].data_ptr(),one_precol[i],64);
        }
		return prebase;
	}
    
    // 获取每个batch上的待编码字符的qv上下文
	at::Tensor& get_preqv() {
		for (uint64_t i = 0; i < num_batch; i++) {
            for (uint64_t j = 0; j < 64; j++) {
                one_precol[i][j]=content_q[i*num_reads_per_batch + precol_index + j];
            }
		}
        for (uint64_t i = 0; i < num_batch; i++){
            memcpy(preqv[i].data_ptr(),one_precol[i],64);
        }
        precol_index++;
		return preqv;
	}

	~input() {
		inputFile.close();
        inputFile_qv.close();
		free(content_b);
        free(content_q);
	}

};



class tempDecoder{
public:

	vector<char*> outputs;
    char *content;
    ifstream inputFile_qv;
	char one_precol[num_max_batch_size][64];
    at::Tensor prebase=torch::zeros({num_max_batch_size, 64},torch::TensorOptions().dtype(torch::kInt8));
    at::Tensor preqv=torch::zeros({num_max_batch_size, 64},torch::TensorOptions().dtype(torch::kInt8));
    uint64_t precol_index = 0;

	tempDecoder(string fileName_qv){
        content = (char*)malloc(num_batch * num_reads_per_batch);
        inputFile_qv.open(fileName_qv);
        char buf_qv[1];
        for (uint64_t i = 0; i < len_reads; i++){
            inputFile_qv.read(buf_qv,1);
            content[i] = buf_qv[0]-' ';
        }
        precol_index = 0;

		outputs.resize(num_batch);

		for(uint64_t i=0;i<num_batch;i++){
			outputs[i]=(char *)malloc(num_reads_per_batch);
		}
	}
	~tempDecoder(){
		for(uint64_t i=0;i<num_batch;i++){
			free(outputs[i]);
		}
        inputFile_qv.close();
		free(content);
	}
	void writeQs(ofstream &out_file){
		for(int i=0;i<num_batch-1;i++){
			for (int j=0;j<num_reads_per_batch;j++){
				out_file<<outputs[i][j];
			}
		}
		uint32_t remin=len_reads - (num_batch-1)*num_reads_per_batch;
        for (int k=0;k<remin;k++){
            out_file<<outputs[num_batch-1][k];
        }

	}
    
    // 获取每个batch上的待编码字符的base上下文
	at::Tensor& get_prebase() {
		for (uint64_t i = 0; i < num_batch; i++) {
            for (int j = 0; j < 64; j++) {
                one_precol[i][j]=char2int[outputs[i][precol_index+j]];
            }
		}
        for (uint64_t i = 0; i < num_batch; i++){
            memcpy(prebase[i].data_ptr(),one_precol[i],64);
        }
		return prebase;
	}
    
    // 获取每个batch上的待编码字符的qv上下文
	at::Tensor& get_preqv() {
		for (uint64_t i = 0; i < num_batch; i++) {
            for (uint64_t j = 0; j < 64; j++) {
                one_precol[i][j]=content[i*num_reads_per_batch + precol_index + j];
            }
		}
        for (uint64_t i = 0; i < num_batch; i++){
            memcpy(preqv[i].data_ptr(),one_precol[i],64);
        }
        precol_index++;
		return preqv;
	}

};

class fileToMem{
public:
	char *binaryMap= nullptr;
	__uint32_t total_size;
	mem *mems= nullptr;
	vector<uint32_t> start_pos;
	uint32_t file_pos;
	ifstream inFile;
	bool refresh(){
		inFile.seekg(file_pos+0);
		inFile.read((char*)&(this->total_size),4);
		if(binaryMap!= nullptr)
			free(binaryMap);
		binaryMap = (char *)malloc(total_size);

		inFile.seekg(file_pos+0);
		inFile.read((char*)binaryMap,total_size);

		inFile.seekg(file_pos+4);
		start_pos.resize(0);
		for(uint64_t i=0;i<num_batch;i++){
			uint32_t tm;
			inFile.read((char*)&tm,4);
			start_pos.push_back(tm);
			mems[i].start=0;
		}
		for(uint64_t i=0;i<num_batch-1;i++){
			mems[i].true_size=start_pos[i+1]-start_pos[i];
			memcpy(mems[i].data,binaryMap+start_pos[i],start_pos[i+1]-start_pos[i]);
		}
		mems[num_batch-1].true_size=total_size-start_pos[num_batch-1];
		memcpy(mems[num_batch-1].data,binaryMap+start_pos[num_batch-1],total_size-start_pos[num_batch-1]);
		file_pos+=this->total_size;
		return inFile.good();
	}
	fileToMem(string fileName) {

		inFile.open(fileName,ios::binary);
		file_pos=0;
		mems=new mem[num_max_batch_size];

	}
	~fileToMem(){
		delete []mems;
		if(binaryMap!= nullptr)
			free(binaryMap);
		inFile.close();
	}
};
class memToFile {
public:
	char *binaryMap;
	__uint32_t true_batch_size;
	__uint32_t total_size;
	uint32_t num_rows;

	vector<uint32_t> start_pos;
	memToFile(uint32_t true_batch_size,vector<uint32_t>& out_size) {
        if(true_batch_size){
            this->true_batch_size=true_batch_size;
            start_pos.resize(true_batch_size);
            start_pos[0]=(true_batch_size+1)*4;
            for (int i = 1; i < true_batch_size; ++i) {
                start_pos[i]=start_pos[i-1]+out_size[i-1];
            }
            this->total_size=start_pos[true_batch_size-1]+out_size[true_batch_size-1];
            binaryMap = (char *)malloc(total_size);
        }

	}

	void writeToMem(mem *mems,vector<uint32_t>& out_size){

		uint32_t *ptr=(uint32_t*)binaryMap;
		*ptr++=total_size;
		for (int i = 0; i < true_batch_size; ++i) {
			*ptr++=start_pos[i];
		}
		unsigned char *cptr=(unsigned char *)ptr;
		for (int i = 0; i < true_batch_size; ++i) {
			memcpy(cptr,mems[i].data,out_size[i]);
			cptr+=out_size[i];
		}
		cout<<"current block size: "<<total_size<<endl;
	}
	void writeToFile(ofstream & outFile){

		outFile.write((char*)binaryMap,total_size);
		outFile.flush();
	}
};


#endif //LSTM_COMPRESSION_MEM_H

