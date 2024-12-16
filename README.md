# RT-DETR: Análise Comparativa com Benchmarks de Detecção de Objetos

Este repositório contém os experimentos realizados para a avaliação do modelo **RT-DETR** na tarefa de detecção de árvores individuais em imagens aéreas de alta resolução. A comparação foi feita com os métodos apresentados no estudo de **Zamboni et al. (2021)**, utilizando múltiplos *backbones* e duas versões da base de dados: original e ajustada.

---

## 📄 **Objetivo**
Avaliar o desempenho do modelo **RT-DETR** em comparação com detectores baseados em *anchors* (ex.: Faster R-CNN, RetinaNet) e livres de *anchors* (ex.: FoveaBox), com foco em:
1. Analisar a **métrica AP50** em diferentes cenários.
2. Comparar o impacto de dados ajustados contendo apenas *bounding boxes* regulares.
3. Explorar a variação dos resultados em múltiplos *folds* da base de dados.

---

## 📊 **Metodologia**

### **1. Dados Utilizados**
- **Base Original**: Imagens rotuladas fornecidas no estudo de Zamboni et al. (2021).
- **Base Ajustada**: Versão com remoção de *bounding boxes* irregulares para melhorar a qualidade das anotações.

### **2. Modelos e Backbones**
Foram testados os seguintes *backbones* no modelo RT-DETR:
- **ResNet**: rtdetr\_r18vd, rtdetr\_r34vd, rtdetr\_r50vd, rtdetr\_r101vd.
- **DLA-34**: rtdetr\_dla34.
- **RegNet**: rtdetr\_regnet.

### **3. Procedimentos de Treinamento**
- **Hiperparâmetros**: Foram replicados os parâmetros padrão do RT-DETR, com exceção do número de épocas.
- **Treinamento**: Realizado em **200 épocas** com monitoramento de estabilização da métrica AP50.
- **Hardware**: Google Colab com GPU NVIDIA A100 (40 GB).

---

## 📈 **Resultados**

### **Desempenho dos Backbones**
A tabela abaixo resume os valores médios de **AP50** obtidos nos testes.

| **Modelo**          | **AP50 Orig (%)** | **AP50 Base Ajustada (%)** | **Melhor Fold (%)** | **Pior Fold (%)** | **Desvio Padrão** |
|---------------------|------------------|---------------------------|---------------------|-------------------|------------------|
| rtdetr\_dla34       | 66.8             | **68.4**                  | **70.5**            | 65.8              | 1.70             |
| rtdetr\_r50vd       | 67.4             | **68.0**                  | **71.4**            | 64.5              | 2.75             |
| rtdetr\_r101vd      | 65.7             | 66.2                      | 67.9                | 63.7              | 2.01             |
| rtdetr\_r18vd       | 64.3             | 64.7                      | 66.1                | 62.8              | **1.20**         |
| rtdetr\_regnet      | 65.4             | 65.9                      | 68.3                | 62.0              | 2.64             |

### **Principais Observações**
1. O *backbone* **DLA-34** obteve a melhor performance geral, alcançando **AP50 de 68.4\%** na base ajustada.
2. A remoção de anotações irregulares trouxe ganhos médios de **1.6 a 2.0 pontos percentuais** na AP50.
3. A variabilidade entre os *folds* foi mais pronunciada nos modelos com *backbone* ResNet-50.

---

## 🚀 **Trabalhos Futuros**
- Avaliar o desempenho do **RT-DETR v2**, que apresenta melhorias promissoras na métrica AP50.
- Realizar análise detalhada de **eficiência computacional**, comparando o tempo de inferência e FLOPs.
- Explorar estratégias de **fine-tuning** para reduzir o número de épocas necessárias no treinamento.
- Testar o RT-DETR em dados multiespectrais e outros cenários de aplicação.

---

## 🔧 **Reproduzindo os Experimentos**

### **Pré-requisitos**
- **Python 3.8+**
- Bibliotecas necessárias: PyTorch, torchvision, numpy, pandas, matplotlib, OpenCV.
- **Google Colab** (para utilização da GPU A100).

### **Abra o arquivo RT-DETR.ipynb** e siga as instruções

🧑‍💻 Contato

Se tiver dúvidas ou sugestões, entre em contato:
- Nome: Rogério Lopes
- Email: rglopes@gmail.com
