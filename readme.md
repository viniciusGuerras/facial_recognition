# Importações e Processamentos de Imagens com FaceNet e ChromaDB

Este projeto implementa um pipeline completo de processamento de imagens faciais, treinamento de rede neural e indexação de embeddings faciais.

O código realiza as seguintes funcionalidades:

1. **Importação de Bibliotecas e Configuração do Ambiente**: instalação e importação de `chromadb`, `torch`, `torchvision`, `pytorch-metric-learning` e outras bibliotecas essenciais. Montagem do Google Drive e extração de arquivos `.zip` contendo imagens.

2. **Dataset Personalizado**: a classe `PathImageFolder` permite carregar imagens mantendo seus caminhos e labels. Transformações aplicadas incluem redimensionamento, recorte aleatório, flip horizontal aleatório, rotação, ajustes de brilho, contraste, saturação e matiz, blur gaussiano e normalização para entrada na rede.

3. **Criação do Modelo**: definição do bloco residual `ResidualBottleneckBlock` com três camadas convolucionais e da arquitetura `FaceNetResNet` baseada em ResNet, que gera embeddings faciais normalizadas de tamanho 512. Inclui pooling adaptativo e camada totalmente conectada para extração de features.

4. **Loop de Treinamento**: treinamento utilizando `TripletMarginLoss` e `MultiSimilarityMiner`, otimizado com `AdamW` e scheduler `ReduceLROnPlateau`. Suporte a GPU, registro da perda média por época e contagem de batches ignorados quando não existem triplets duros.

5. **Geração e Armazenamento de Embeddings na ChromaDB**: extração de embeddings para todas as imagens do dataset, criação de coleção `faces` na ChromaDB e indexação em batches de embeddings, labels e caminhos das imagens. Configuração de batch (`chromas_batch_size = 5000`) para evitar sobrecarga de memória.

**Tecnologias e Bibliotecas Utilizadas**: PyTorch, Torchvision, Pytorch Metric Learning, ChromaDB, Google Colab e Pillow.

