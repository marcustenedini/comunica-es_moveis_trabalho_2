import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


class GeradorInversaNumerica:
    """
    Gera Variáveis Aleatórias a partir de uma PDF arbitrária
    usando Interpolação Spline da CDF Inversa (Método Estável).
    """
    
    def __init__(self, funcao_pdf, x_max=5.0, num_pontos=2000):
        self.funcao_pdf = funcao_pdf
        self.x_max = x_max
        self.num_pontos = num_pontos # Mais pontos = maior precisão
        self.funcao_inversa = None
        
        self._preparar_interpolacao()

    def _preparar_interpolacao(self):
        # 1. Discretizar o domínio X
        x_vals = np.linspace(1e-6, self.x_max, self.num_pontos)
        
        # 2. Calcular a PDF
        pdf_vals = np.array([self.funcao_pdf(x) for x in x_vals])
        
        # 3. Integrar para obter a CDF
        cdf_vals = cumulative_trapezoid(pdf_vals, x_vals, initial=0)
        cdf_vals = cdf_vals / cdf_vals[-1] # Normalizar para max = 1.0
        
        # 4. Remover possíveis duplicatas numéricas na CDF para a interpolação não quebrar
        _, indices_unicos = np.unique(cdf_vals, return_index=True)
        cdf_vals_unicos = cdf_vals[indices_unicos]
        x_vals_unicos = x_vals[indices_unicos]
        
        # 5. Criar a função de interpolação inversa (Eixo X = CDF, Eixo Y = Valores de H)
        # bounds_error=False permite lidar graciosamente com valores ligeiramente fora do limite
        self.funcao_inversa = interp1d(
            cdf_vals_unicos, 
            x_vals_unicos, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(0.0, self.x_max)
        )

    def gerar_amostras(self, num_amostras: int) -> np.ndarray:
        if self.funcao_inversa is None:
            raise ValueError("A função de interpolação não foi inicializada.")
            
        # Sorteia U ~ Uniforme(0, 1)
        U = np.random.rand(num_amostras)
        
        # Passa os valores pela função de interpolação
        amostras_H = self.funcao_inversa(U)
        
        return amostras_H
