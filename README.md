
We want to know if an image has an X or an O.
| X | O |
|-|-|
| ![](img/xo/x.png) | ![](img/xo/x.png) | 


# Kernels, Feature Maps, and Poolimg
## An X has diagnoal lines

| Input | Kernel | Feature Map | Max Pooling | Mean Pooling |
|-|-|-|-|-|
| ![](img/xo/x.png) | ![](img/xo/kernel_back_diagonal.png)| ![](img/xo/x.kernel_back_diagonal.png)| 3.0 | 1.67 |
| ![](img/xo/x.png) | ![](img/xo/kernel_fwd_diagonal.png) | ![](img/xo/x.kernel_fwd_diagonal.png) | 3.0 | 1.67 |

## An O does not have diagnoal lines

| Input | Kernel | Feature Map | Max Pooling | Mean Pooling |
|-|-|-|-|-|
| ![](img/xo/o.png) | ![](img/xo/kernel_back_diagonal.png)| ![](img/xo/o.kernel_back_diagonal.png)| 2.0 | 0.83 |
| ![](img/xo/o.png) | ![](img/xo/kernel_fwd_diagonal.png) | ![](img/xo/o.kernel_fwd_diagonal.png) | 2.0 | 0.83 |

## An O has curves.
| Input | Kernel | Feature Map | Max Pooling | Mean Pooling |
|-|-|-|-|-|
| ![](img/xo/o.png) | ![](img/xo/kernel_topl_curve.png)| ![](img/xo/o.kernel_topl_curve.png) | 4.0 | 1.4 |
| ![](img/xo/o.png) | ![](img/xo/kernel_botr_curve.png) | ![](img/xo/o.kernel_botr_curve.png)| 4.0 | 1.4 |


## An X does not have curves.
| Input | Kernel | Feature Map | Max Pooling | Mean Pooling |
|-|-|-|-|-|
| ![](img/xo/x.png) | ![](img/xo/kernel_topl_curve.png)| ![](img/xo/o.kernel_topl_curve.png) | 3.0 | 1.3 |
| ![](img/xo/x.png) | ![](img/xo/kernel_botr_curve.png) | ![](img/xo/o.kernel_botr_curve.png)| 3.0 | 1.3 |

## X has a set pixel in the corner.
| Input | Kernel | Feature Map | Max Pooling | Mean Pooling |
|-|-|-|-|-|
| ![](img/xo/x.png) | ![](img/xo/kernel_set_corner.png)| ![](img/xo/o.kernel_set_corner.png) | 1.0 | 0.33 |
| ![](img/xo/o.png) | ![](img/xo/kernel_set_corner.png) | ![](img/xo/o.kernel_set_corner.png)| 1.0 | 0.4 |




