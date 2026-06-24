# Modelagem de Canal para Comunicações Móveis
**Análise Matemática, Estatística e Desempenho**

Este repositório contém a implementação em Python para a simulação e análise teórica de canais de comunicação sem fio, baseando-se no modelo generalizado de desvanecimento $\kappa$-$\mu$ sombreado integrado à atenuação geométrica de percurso (Path Loss).

**Autor:** Marcus Emanuel Carvalho Tenedini de Freitas (Universidade de Brasília - UnB)

---

##  Arquitetura do Projeto

O código foi refatorado para garantir a separação de responsabilidades (física, estatística e visualização):

* **`params.py` (O Motor Matemático):** Define a classe `Params` que encapsula as variáveis ambientais ($\kappa$, $\mu$, $m_d$, $\alpha$, $R_M$, $h_a$). Contém as formulações teóricas exatas (PDFs) e as funções de cálculo limite para OP (Outage Probability), SEP (Symbol Error Probability) e CE (Ergodic Capacity).
* **`gerador_inversa.py` (O Motor Estocástico):** Implementa o método da Transformada Inversa Numérica para sortear as amostras do canal (envelope $H$), garantindo aderência absoluta à Função Densidade de Probabilidade do canal simulado.
* **`utils.py` (Orquestração e Plotagem):** Contém as funções de simulação de Monte Carlo (`simular_metricas`) e o gerador de gráficos padronizados (`plotar_curva`).
* **`main.py`:** O script orquestrador central. Executa de forma legível e sequencial todos os gráficos gerados para a apresentação final.

---

##  O Motor Matemático (`params.py`)

O arquivo `params.py` é o núcleo de cálculo do projeto. Ele atua como um repositório central contendo os parâmetros da rede (geometria e potência) e as propriedades estatísticas do canal (modelo $\kappa$-$\mu$ sombreado). Além de armazenar o estado do sistema, a classe `Params` disponibiliza métodos para avaliar as métricas de desempenho tanto de forma teórica (soluções fechadas) quanto para apoiar a simulação de Monte Carlo.

###  A Tática de Estabilidade Numérica (O Domínio Logarítmico)

Um dos maiores desafios ao implementar distribuições estatísticas generalizadas é a presença de funções Fatoriais e funções Gama nas expansões em séries infinitas. Esses valores explodem rapidamente, causando erros de *Overflow* (`Inf`) ou *Underflow* (`0.0` ou `NaN`) nas bibliotecas matemáticas do Python.

Para contornar esse limite da arquitetura da linguagem, o código utiliza a **tática de computação no domínio logarítmico** (implementada em métodos como `_log_coeficiente_Ad`):

* Em vez de calcular `A / B` diretamente (onde `A` e `B` são números gigantescos), o sistema calcula `ln(A) - ln(B)`.
* Fatoriais são substituídos por `gammaln` (o logaritmo natural da função Gama).
* Os valores só retornam ao domínio linear (`np.exp(...)`) no exato momento da soma final da série, quando a magnitude resultante já foi mitigada pela divisão logarítmica. Isso garante estabilidade absoluta, mesmo para matrizes de simulação gigantescas com alta ordem de diversidade.

###  Como a SNR Instantânea é Calculada

A classe simula a Relação Sinal-Ruído (SNR) instantânea de forma fisicamente coerente, separando a degradação do sinal em etapas sequenciais no método `SNR()`:

1.  **Potência de Transmissão (Base):** O ponto de partida é a potência de transmissão ajustada pelo ruído basal do hardware.
2.  **Atenuação de Larga Escala (Path Loss):** A potência base é multiplicada pela perda geométrica determinística (`rho_t`), que absorve a energia do sinal com base na distância 3D exata do usuário até a antena e no expoente de atenuação da cidade ($\alpha$).
3.  **Desvanecimento de Pequena Escala (Fading):** O que sobra do sinal sofre a interferência das múltiplas trajetórias e bloqueios urbanos. Isso é injetado multiplicando o sinal restante pela variável aleatória do ganho do canal ao quadrado ($H^2$).

---

###  Dicionário de Métodos da Classe `Params`

#### Geometria e Espaço
* `gerar_R(num_users)`: Distribui usuários uniformemente na área de uma microcélula circular. Utiliza a raiz quadrada da variável aleatória para evitar uma concentração irreal de pontos no centro da célula.
* `calcular_D_R()`: Converte a distância radial plana (2D) em distância euclidiana (3D), calculando a hipotenusa gerada pela diferença de altura entre a antena da torre e o dispositivo do usuário.

#### Física do Canal
* `rho_t(D_R)`: Calcula o coeficiente de atenuação de percurso (Path Loss) baseado na distância real e na frequência de operação da rede.
* `recieved_signal(...)`: Reconstrói a equação física da onda eletromagnética recebida, somando o ruído branco gaussiano (AWGN) ao sinal degradado pelo desvanecimento.
* `SNR(amostras_H, D_R)`: Retorna a Relação Sinal-Ruído instantânea ($\Gamma$) que o hardware do dispositivo efetivamente processaria após as perdas geométricas e estocásticas.
* `calcular_gamma_0(gamma_t_linear)`: Determina a SNR média teórica de referência no limite seguro de distância da antena (geralmente a 1 metro de distância).

#### Soluções Analíticas (Avaliação Teórica)
* `_log_coeficiente_Ad(n)` / `_coeficiente_Ad(n)`: Funções auxiliares que geram as constantes ponderadas para as expansões em séries de Maclaurin exigidas pelas equações do modelo, operando no domínio logarítmico para garantir segurança contra estouro de memória.
* `pdf_fHx(x, max_terms)`: Avalia a curva de Densidade de Probabilidade do canal. Possui um mecanismo de interrupção dinâmica que cessa o cálculo da série infinita assim que os termos convergem para valores irrisórios (`< 1e-12`), otimizando o processamento.
* `calcular_OP_teorica(...)`: Avalia a fórmula fechada da Probabilidade de Outage (OP), relacionando os limites geométricos espaciais com a Função Gama Incompleta.
* `calcular_ASEP_teorica(...)`: Calcula a Probabilidade Média de Erro de Símbolo (SEP) através de integrais exatas que invocam Funções Hipergeométricas de Gauss (`hyp2f1`).
* `calcular_CE_teorica(...)`: Retorna o teto da Capacidade Ergódica (CE) aplicando as integrais resolvidas via Função G de Meijer (`mpmath.meijerg`).
* `calcular_CE_integracao(...)`: Abordagem de contingência matemática. Caso a formulação de Meijer-G falhe por instabilidade de convergência (comum em sombreamentos muito brandos), este método extrai a capacidade integrando numericamente a Probabilidade de Outage acumulada.

##  O Motor Estocástico (`gerador_inversa.py`)

O arquivo `gerador_inversa.py` é o responsável pela simulação de Monte Carlo. Ele implementa a classe `GeradorInversaNumerica`, que funciona como um motor estocástico genérico. Seu objetivo é gerar milhões de amostras aleatórias de ganho de canal (envelope $H$) que obedeçam rigorosamente à curva de Densidade de Probabilidade (PDF) definida pela física do sistema.

### Táticas de Engenharia e Estabilidade Numérica

A geração de variáveis aleatórias para canais com desvanecimento generalizado (como o $\kappa$-$\mu$ sombreado) apresenta um problema computacional grave: a Função de Distribuição Acumulada (CDF) desses modelos não possui uma função inversa em forma fechada (analítica). Para resolver isso com alta performance, a classe adota as seguintes estratégias:

#### 1. Transformada Inversa Numérica
Em vez de tentar inverter equações complexas com funções hipergeométricas, o gerador recebe a equação da PDF como uma "caixa preta" e constrói a CDF numericamente. 
* O domínio do sinal é discretizado e a função `cumulative_trapezoid` do SciPy calcula a área sob a curva (integral) passo a passo. 
* O resultado é normalizado forçadamente para garantir que a probabilidade máxima feche em exatamente `1.0`, corrigindo eventuais resíduos de arredondamento de máquina (ponto flutuante).

#### 2. Prevenção de Quebras
Para que a técnica da transformada inversa funcione, a CDF precisa ser invertida (o eixo Y vira o eixo X para a interpolação). O módulo `interp1d` do SciPy exige que o novo eixo X seja estritamente crescente. 
* **O Problema:** Em áreas onde a PDF é virtualmente zero (caudas muito longas), a CDF estagna e gera valores repetidos. Isso causaria um erro fatal de interpolação.
* **A Solução:** O uso da instrução `np.unique` filtra a matriz da CDF, retirando degraus repetidos e garantindo uma curva perfeitamente monotônica e contínua, blindando o código contra falhas de execução.

#### 3. Vetorização Extrema (Alta Performance)
A preparação matemática (`_preparar_interpolacao`) é um processo custoso, ele ocorre uma vez durante a inicialização da classe (`__init__`). Uma vez que a função interpoladora contínua é criada na memória, o método de sorteio (`gerar_amostras`) utiliza operações puramente vetorizadas do NumPy para injetar matrizes com milhões de números uniformes e convertê-los em amostras de canal válidas.

---

### Dicionário de Métodos da Classe `GeradorInversaNumerica`

* `__init__(funcao_pdf, x_max, num_pontos)`: Construtor que recebe a função teórica do canal e a resolução desejada. Quanto maior o número de pontos, mais fiel será a cauda da distribuição, ideal para capturar os desvanecimentos profundos (*deep fades*).
* `_preparar_interpolacao()`: Método privado que orquestra a conversão da PDF em uma função interpoladora inversa. Ele discretiza o eixo, integra numericamente via regra dos trapézios, higieniza as duplicatas da matriz e constrói o mapeador estatístico contínuo.
* `gerar_amostras(num_amostras)`: Método público. Ele sorteia variáveis uniformes $U \sim (0, 1)$ e as injeta na função inversa pré-computada, devolvendo um array NumPy populado com os ganhos instantâneos do canal ($H$), prontos para serem multiplicados pela potência da antena.
##  Como Executar

Instale as dependências:
```bash
pip install numpy scipy matplotlib mpmath
