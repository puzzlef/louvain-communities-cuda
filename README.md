Design of CUDA-based [Louvain algorithm] for [community detection].

Community detection involves identifying natural divisions in networks, a crucial task for many large-scale applications. This report presents [GVE-Louvain], one of the most efficient multicore implementations of the Louvain algorithm, a high-quality method for community detection. Running on a dual 16-core Intel Xeon Gold 6226R server, GVE-Louvain outperforms Vite, Grappolo, NetworKit Louvain, and cuGraph Louvain (on an NVIDIA A100 GPU) by factors of `50x`, `22x`, `20x`, and `5.8x`, respectively, achieving a processing rate of `560M` edges per second on a `3.8B`-edge graph. Additionally, it scales efficiently, improving performance by `1.6x` for every thread doubling. The paper also presents **ν-Louvain**, a GPU-based implementation. When evaluated on an NVIDIA A100 GPU, **ν-Louvain** performs only on par with GVE-Louvain, largely due to reduced workload and parallelism in later algorithmic passes. These results suggest that CPUs, with their flexibility in handling irregular workloads, may be better suited for community detection tasks.

<br>


Below we plot the time taken by [Grappolo] (Louvain), [NetworKit] Louvain, [Nido] (Louvain), [cuGraph] Louvain, and **ν-Louvain** on 13 different graphs. **ν-Louvain** surpasses Grappolo, NetworKit Louvain, Nido, and cuGraph Louvain by `20×`, `17×`, `61×`, and `5.0×` respectively, achieving a processing rate of `405M` edges/s on a `2.2𝐵` edge graph.

[![](https://github.com/user-attachments/assets/51ac7bb6-7bfd-4919-9169-df11eab2383e)][sheets-o1]

Below we plot the speedup of **ν-Louvain** wrt Grappolo, NetworKit Louvain, Nido, and cuGraph Louvain.

[![](https://github.com/user-attachments/assets/0cbe60a2-d86d-457c-89e0-e64eef020213)][sheets-o1]

Finally, we plot the modularity of communities identified by Grappolo, NetworKit Louvain, Nido, cuGraph Louvain, and **ν-Louvain**. **ν-Louvain** on average obtains `1.1%`, `1.2%`, and `1.3%` lower modularity than Grappolo, NetworKit
Louvain, and cuGraph Louvain (where cuGraph Louvain runs), but `45%` higher modularity than Nido.

[![](https://github.com/user-attachments/assets/de1170b7-5781-498a-b8c4-7b5ed157c4fa)][sheets-o1]

Refer to our technical report for more details: \
[CPU vs. GPU for Community Detection: Performance Insights from GVE-Louvain and ν-Louvain][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.

[GVE-Louvain]: https://arxiv.org/abs/2312.04876
[Grappolo]: https://github.com/ECP-ExaGraph/grappolo
[NetworKit]: https://github.com/networkit/networkit
[Nido]: https://github.com/sg0/nido
[cuGraph]: https://github.com/rapidsai/cugraph
[Louvain algorithm]: https://en.wikipedia.org/wiki/Louvain_method
[community detection]: https://en.wikipedia.org/wiki/Community_search
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[sheets-o1]: https://docs.google.com/spreadsheets/d/1eXc7hkjuEAoIGLagzQrNQF_YubvdUiupgqJHPNxxFnk/edit?usp=sharing
[report]: https://arxiv.org/abs/2501.19004

<br>
<br>


### Code structure

The code structure of **ν-Louvain** is as follows:

```bash
- inc/_*.hxx: Utility functions
- inc/**.hxx: Common graph functions
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/Graph.hxx: Graph data structure
- inc/louvian.hxx: OpenMP-based Louvian algorithm
- inc/louvainCuda.hxx: CUDA-based Louvian algorithm
- inc/hashtableCuda.hxx: CUDA-based hybrid quadratic-double hashtable
- inc/mtx.hxx: Graph (MTX format) file reader
- inc/properties.hxx: Graph property functions (e.g., modularity)
- inc/symmetrize.hxx: Make graph symmetric
- inc/update.hxx: Graph update functions
- inc/main.hxx: Main header
- main.cxx: Experimentation code
- main.sh: Experimentation script
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

<br>
<br>


## References

- [Fast unfolding of communities in large networks; Vincent D. Blondel et al. (2008)](https://arxiv.org/abs/0803.0476)
- [Community Detection on the GPU; Md. Naim et al. (2017)](https://arxiv.org/abs/1305.2006)
- [Scalable Static and Dynamic Community Detection Using Grappolo; Mahantesh Halappanavar et al. (2017)](https://ieeexplore.ieee.org/document/8091047)
- [From Louvain to Leiden: guaranteeing well-connected communities; V.A. Traag et al. (2019)](https://www.nature.com/articles/s41598-019-41695-z)
- [CS224W: Machine Learning with Graphs | Louvain Algorithm; Jure Leskovec (2021)](https://www.youtube.com/watch?v=0zuiLBOIcsw)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [Fetch-and-add using OpenMP atomic operations](https://stackoverflow.com/a/7918281/1413259)

<br>
<br>


[![](https://img.youtube.com/vi/yqO7wVBTuLw/maxresdefault.jpg)](https://www.youtube.com/watch?v=yqO7wVBTuLw)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/louvain-communities-cuda)
