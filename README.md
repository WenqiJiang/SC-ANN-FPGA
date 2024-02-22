# FPGA-accelerated Vector Search

This is the repository of our SC'23 paper titled [Co-design Hardware and Algorithm for Vector Search](https://dl.acm.org/doi/pdf/10.1145/3581784.3607045). We built FPGA accelerators for product-quantization-based vector search.

## Abstract

Vector search has emerged as the foundation for large-scale information retrieval and machine learning systems, with search engines like Google and Bing processing tens of thousands of queries per second on petabyte-scale document datasets by evaluating vector similarities between encoded query texts and web documents. As performance demands for vector search systems surge, accelerated hardware offers a promising solution in the post-Moore’s Law era. We introduce FANNS, an end-to-end and scalable vector search framework on FPGAs. Given a user-provided recall requirement on a dataset and a hardware resource budget, FANNS automatically co-designs hardware and algorithm, subsequently generating the corresponding accelerator. The framework also supports scale-out by incorporating a hardware TCP/IP stack in the accelerator. FANNS attains up to 23.0× and 37.2× speedup compared to FPGA and CPU baselines, respectively, and demonstrates superior scalability to GPUs, achieving 5.5× and 7.6× speedup in median and 95th percentile (P95) latency within an eight-accelerator configuration. The remarkable performance of FANNS lays a robust groundwork for future FPGA integration in data centers and AI supercomputers.

## Citation

```
@inproceedings{jiang2023co,
  title={Co-design hardware and algorithm for vector search},
  author={Jiang, Wenqi and Li, Shigang and Zhu, Yu and de Fine Licht, Johannes and He, Zhenhao and Shi, Runbin and Renggli, Cedric and Zhang, Shuai and Rekatsinas, Theodoros and Hoefler, Torsten and others},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--15},
  year={2023}
}
```
