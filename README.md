
We want to know if an image has an X or an O.
| X | O |
|-|-|
| ![](img/xo/x.png) | ![](img/xo/x.png) | 


An X has diagnoal lines

| Input | Kernel | Feature Map | max | mean |
|-|-|-|-|-|
| ![](img/xo/x.png) | ![](img/xo/kernel_back_diagonal.png)| ![](img/xo/x.kernel_back_diagonal.png)| 3.0 | 1.67 |
| ![](img/xo/x.png) | ![](img/xo/kernel_fwd_diagonal.png) | ![](img/xo/x.kernel_fwd_diagonal.png) | 3.0 | 1.67 |

An O does not have diagnoal lines

| Input | Kernel | Feature Map | max | mean |
|-|-|-|-|-|
| ![](img/xo/o.png) | ![](img/xo/kernel_back_diagonal.png)| ![](img/xo/o.kernel_back_diagonal.png)| 2.0 | 0.83 |
| ![](img/xo/o.png) | ![](img/xo/kernel_fwd_diagonal.png) | ![](img/xo/o.kernel_fwd_diagonal.png) | 2.0 | 0.83 |
