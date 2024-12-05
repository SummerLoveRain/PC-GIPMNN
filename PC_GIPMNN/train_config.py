import numpy as np

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([0, 0])
ub = np.array([170, 170])

# sigmaA1 = 0.4;
# sigmaA2 = 0.4;
# sigmaA3 = 0.38;
# sigmaA4 = 0.01;
# sigmaS1 = 0.53;
# sigmaS2 = 0.53;
# sigmaS3 = 0.2;
# sigmaS4 = 0.89;
# vSigmaF1 = 0.085;
# vSigmaF2 = 0.085;
# vSigmaF3 = 0.0;
# vSigmaF4 = 0.0;

# sigmaA1 = 0.3;
# sigmaA2 = 0.4;
# sigmaA3 = 0.38;
# sigmaA4 = 0.01;
# sigmaS1 = 0.45;
# sigmaS2 = 0.53;
# sigmaS3 = 0.2;
# sigmaS4 = 0.89;
# vSigmaF1 = 0.055;
# vSigmaF2 = 0.085;
# vSigmaF3 = 0.0;
# vSigmaF4 = 0.0;

sigmaA1 = 2.0;
sigmaA2 = 0.087;
sigmaA3 = 0.38;
sigmaA4 = 0.01;
sigmaS1 = 0.53;
sigmaS2 = 0.55;
sigmaS3 = 0.2;
sigmaS4 = 0.89;
vSigmaF1 = 0.079;
vSigmaF2 = 0.085;
vSigmaF3 = 0.0;
vSigmaF4 = 0.0;

D1 = 1/(3*(sigmaA1+sigmaS1));
D2 = 1/(3*(sigmaA2+sigmaS2));
D3 = 1/(3*(sigmaA3+sigmaS3));
D4 = 1/(3*(sigmaA4+sigmaS4));
