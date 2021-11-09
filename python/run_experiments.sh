#/bin/bash
model_dir="../data/trained_models"
compressed_dir="../data/compressed"
data_dir="../data/processed_files"
logs_dir="../data/logs_data"
original_dir="../data/files_to_be_compressed"
export CUDA_VISIBLE_DEVICES=$2

mkdir -p $data_dir
f="$original_dir/$3.fastq"
output_file="$data_dir/$3"
# 字典参数，et.'A':'0'
params_file="$data_dir/$3.stat.info"
# 从fastq中提取base.npz/qv.npz
python extract_bq.py $f $output_file $params_file
fb_npz="$data_dir/$3.base.npz"
fb_txt="$data_dir/$3.base.txt"
fq_npz="$data_dir/$3.qv.npz"
fq_txt="$data_dir/$3.qv.txt"

# Storing the model
model_file_temp="$model_dir/$3/model_net.pth"
model_file="$model_dir/$3/model.pt"
log_file="$logs_dir/$3/log.csv"
output_dir="$compressed_dir/$3"
recon_file_name="$output_dir/$3.rec.txt"
output_prefix="$output_dir/$3.bin"
mkdir -p "$model_dir/$3"
mkdir -p "$logs_dir/$3"
mkdir -p $output_dir

status=$?
echo $status

if cmp $recon_file_name "$original_dir/$basename.txt"; then
    continue 
else
    echo "continuing"
fi

echo "Starting training ..." | tee -a $log_file

# 利用待压缩文件作为训练数据，训练网络将参数存入指定文件
python trainer.py -d $fb_npz -dq $fq_npz -rate $1 -data_params $params_file -model $model_file_temp -log_file $log_file

python dumpModel.py -data_params $params_file -model $model_file_temp -model_pt $model_file

# Perform Compression
echo "Starting Compression ..." | tee -a $log_file
../c++/DeepQC c $params_file $model_file $fb_txt $fq_txt $output_prefix $4 2>/dev/null
../c++/DeepQC x $params_file $model_file $output_prefix $fq_txt $recon_file_name $4 2>/dev/null
cmp $recon_file_name "$original_dir/$basename.txt" >> $log_file
#echo "- - - - - "



