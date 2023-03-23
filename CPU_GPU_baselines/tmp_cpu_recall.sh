DBNAME='SIFT100M'

# IVF, no OPQ
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF1024,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF2048,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF4096,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF8192,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF16384,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF32768,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF65536,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key IVF65536,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
for i in '10','70' ;
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF131072,PQ16 --recall_goal $RECALL --topK $TOPK 
done
for i in '10','70' ;
do
    IFS=',' read TOPK RECALL <<< "${i}"
    python bench_cpu_recall.py --dbname $DBNAME --index_key IVF262144,PQ16 --recall_goal $RECALL --topK $TOPK 
done

# IVF, with OPQ
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF1024,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF2048,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF4096,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF8192,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF16384,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF32768,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF65536,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF131072,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
# for i in '10','70' ;
# do
#     IFS=',' read TOPK RECALL <<< "${i}"
#     python bench_cpu_recall.py --dbname $DBNAME --index_key OPQ16,IVF262144,PQ16 --recall_goal $RECALL --topK $TOPK 
# done
