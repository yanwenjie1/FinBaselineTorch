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
