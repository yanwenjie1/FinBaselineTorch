# FinBaselineTorch
NLP 模型 Torch实现
# 项目结构
- **data**
  - 数据集
    - ***preprocess.py*** 构造输入
- **utils**
  - ***utils.py*** 工具箱
  - ***bert_base_model.py*** bert类预训练模型加载
  - ***bert_ner_model.py*** seq2seq基本模型
  - ***bert_classify_model.py*** seq2one基本模型
  - ***adversarial_training.py*** 对抗学习实现
  - ***adversarial_example.ipynb*** 对抗学习示例
- **config.py** 配置文件
- **train.py** 训练文件
- **server.py** 接口文件
- **test.py** 测试文件
# 环境配置
- step1 基础环境确认
  - win+R 运行cmd 进入命令行界面
  - 输入 nvidia-smi 查看显卡驱动是否支持11.3版本cuda （保证CUDA Version大于11.3即可）
  - win+R 运行cmd 进入命令行界面 输入conda info -e 确认anaconda/miniconda安装成功
  - 建议配置conda源 参考 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
  - 建议配置pip源   参考 https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
