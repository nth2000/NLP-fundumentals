# Lecuture11 Question Answering

## QA的定义

建立系统**自动回答人类以自然语言提出的问题**

- 信息来源：文本，网页文档，知识库，表格，图片 etc
- 问题类型：事实性问题 vs 非事实性问题，开放域问题 vs 封闭域问题，简单 vs 组合问题
- 回答类型：文本的一段，一段话，一个表格，或yes/no

**IBM Watson的系统架构**

![image-20220727143153558](Lecuture11 Question Answering.assets/image-20220727143153558.png)

包含四个高度模块化的架构。

许多SOTA QA系统都建立在端到端训练和预训练模型之上

![image-20220727143411752](Lecuture11 Question Answering.assets/image-20220727143411752.png)

**knowledge-based QA**

以大型数据库为基础回答问题。

将问题转换成某种逻辑形式，逻辑形式在DATABASE中执行来给出最终结果：

![image-20220727143813912](Lecuture11 Question Answering.assets/image-20220727143813912.png)

**visual QA**

根据图像回答问题。需要理解问题文本和理解图像

## Reading Comprehension

