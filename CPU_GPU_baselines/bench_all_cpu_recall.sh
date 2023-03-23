DBNAME='Deep1000M'

#  1','20' '1','25' '1','30' '10','40' '10','60' '10','80' '100','60' '100','80' '100','95'

# IVF, no OPQ
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF1024,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF2048,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF4096,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF8192,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF16384,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF32768,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF65536,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF131072,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF262144,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done

# IVF, with OPQ
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF1024,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF2048,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF4096,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF8192,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF16384,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF32768,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
for i in '1','25'  '10','70' '100','90';
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF65536,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF131072,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF262144,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done

# IMI, no OPQ
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x8,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x9,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x10,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x11,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x12,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x13,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
# for i in '1','25'  '10','70' '100','90';
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IMI2x14,PQ16 --recall_goal $RECALL --topK $TOPK >> cpu_recall_tmp
# done
