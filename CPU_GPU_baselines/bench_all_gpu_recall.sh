DBNAME='SIFT1000M'
NGPU='3'
STARTGPU='3'

# IVF, no OPQ
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF1024,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF2048,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF4096,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF8192,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF16384,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF32768,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF65536,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF131072,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key IVF262144,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done

for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF1024,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF2048,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF4096,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF8192,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF16384,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF32768,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF65536,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF131072,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done
for i in '1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_gpu_1bn.py -dbname $DBNAME -index_key OPQ16,IVF262144,PQ16 -recall_goal $RECALL -topK $TOPK -ngpu $NGPU -startgpu $STARTGPU -qbs 512 >> gpu_recall
done



