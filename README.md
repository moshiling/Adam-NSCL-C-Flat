# Adam-NSCL-C-Flat
这次不是把 C-Flat 当成一个新 optimizer 直接替换 Adam-NSCL，而是把它放在“梯度生成前端”：先用 C-Flat 生成更平坦的梯度，再把这个梯度交给 Adam-NSCL 原本的 Adam + SVD/null-space + regularization 更新链条。后面的实验又说明，全模型、宽 scope、带较强 g1 的版本都不太适合 CL，所以路线逐步收敛成了“last_block selective + very small lambda，甚至 g0-only”。
