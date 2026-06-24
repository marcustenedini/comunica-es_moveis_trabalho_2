import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import os
from gerador_inversa import GeradorInversaNumerica
from params import Params

def preparar_diretorios():
    os.makedirs('images', exist_ok=True)
    os.makedirs('data', exist_ok=True)

def simular_metricas(sistema: Params, num_amostras: int, snr_db_grid: np.ndarray, gamma_th: float = 1.0):
    gerador = GeradorInversaNumerica(funcao_pdf=sistema.pdf_fHx)
    amostras_h = gerador.gerar_amostras(num_amostras)
    
    distancias_r = sistema.gerar_R(num_amostras) 
    distancias_d_r = sistema.calcular_D_R()

    path_loss = sistema.rho_t(distancias_d_r)
    canal = path_loss * (amostras_h ** 2)

    resultados_op = []
    resultados_sep = []
    resultados_ce = []

    for snr_db in snr_db_grid:
        gamma_instante = (10 ** (snr_db / 10)) * canal
        resultados_op.append(np.mean(gamma_instante < gamma_th))
        resultados_sep.append(np.mean(0.5 * erfc(np.sqrt(gamma_instante))))
        resultados_ce.append(np.mean(np.log2(1 + gamma_instante)))

    return resultados_op, resultados_sep, resultados_ce

def plotar_curva(x_vals, curvas, titulo, ylabel, escala_y='log', arquivo=None, y_lim=None):
    ciclo_cores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(9, 6))

    cor_idx = 0
    simulado_ja_na_legenda = False # Flag para controlar a legenda da simulação

    for valores, label in curvas:
        cor_atual = ciclo_cores[cor_idx % len(ciclo_cores)]
        
        if 'teór' in label.lower() or 'teor' in label.lower():
            estilo = {'linewidth': 2, 'linestyle': '-', 'marker': 'None', 'color': cor_atual}
            label_plot = label # Mantém a legenda teórica inalterada (com os parâmetros)
        else:
            estilo = {
                'linestyle': 'None', 'marker': '+', 'markersize': 5, 
                'markevery': 3, 'fillstyle': 'none', 'color': 'black'
            }
            # Adiciona os marcadores à legenda apenas na primeira vez
            if not simulado_ja_na_legenda:
                label_plot = 'Simulação' # Nome genérico e limpo
                simulado_ja_na_legenda = True
            else:
                label_plot = '_nolegend_' # Instrui o Matplotlib a ignorar este item na legenda
            
            cor_idx += 1 

        if escala_y == 'log':
            plt.semilogy(x_vals, valores, label=label_plot, **estilo)
        else:
            plt.plot(x_vals, valores, label=label_plot, **estilo)

    plt.title(titulo)
    plt.xlabel(r'Transmit SNR, $\gamma_t$ (dB)')
    plt.ylabel(ylabel)
    if y_lim is not None: plt.ylim(*y_lim)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if arquivo is not None:
        plt.savefig(arquivo, dpi=300, bbox_inches='tight')
    plt.show()