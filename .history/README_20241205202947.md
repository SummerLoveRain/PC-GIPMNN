# [Physics-constrained neural network for solving discontinuous interface K-eigenvalue problem with application to reactor physics](https://github.com/SummerLoveRain/PC-GIPMNN)

Machine learning-based modeling of reactor physics problems has attracted increasing interest in recent years. Despite some progress in one-dimensional problems, there is still a paucity of benchmark studies that are easy to solve using traditional numerical methods albeit still challenging using neural networks for a wide range of practical problems. We present two networks, namely the Generalized Inverse Power Method Neural Network (GIPMNN) and Physics-Constrained GIPMNN (PC-GIPIMNN) to solve K-eigenvalue problems in neutron diffusion theory. GIPMNN follows the main idea of the inverse power method and determines the lowest eigenvalue using an iterative method. The PC-GIPMNN additionally enforces conservative interface conditions for the neutron flux. Meanwhile, Deep Ritz Method (DRM) directly solves the smallest eigenvalue by minimizing the eigenvalue in Rayleigh quotient form. A comprehensive study was conducted using GIPMNN, PC-GIPMNN, and DRM to solve problems of complex spatial geometry with variant material domains from the field of nuclear reactor physics. The methods were compared with the standard finite element method. The applicability and accuracy of the methods are reported and indicate that PC-GIPMNN outperforms GIPMNN and DRM.

For more information, please refer to the following: (https://doi.org/10.1007/s41365-023-01313-0)

## Citation

    @article{yang2023physics,
    title={Physics-constrained neural network for solving discontinuous interface K-eigenvalue problem with application to reactor physics},
    author={Yang, Qi-Hong and Yang, Yu and Deng, Yang-Tao and He, Qiao-Lin and Gong, He-Lin and Zhang, Shi-Quan},
    journal={Nuclear Science and Techniques},
    volume={34},
    number={10},
    pages={161},
    year={2023},
    publisher={Springer}
    }
