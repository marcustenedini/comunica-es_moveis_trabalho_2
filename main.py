import numpy as np
import matplotlib.pyplot as plt

from params import Params
from utils import simular_metricas, plotar_curva, preparar_diretorios
from gerador_inversa import GeradorInversaNumerica

# ====================================================================
# CONFIGURAÇÕES GLOBAIS DE SIMULAÇÃO
# ====================================================================
NUM_AMOSTRAS_OP_SEP = 1_000_000
NUM_AMOSTRAS_CE = 1_000_000
NUM_AMOSTRAS_PDF = 1_000_000
SNR_GRID = np.arange(65, 111, 1)
SNR_GRID_CE = np.arange(65, 111, 1)

# ====================================================================
# MÓDULOS DE PLOTAGEM (ESTATÍSTICA, ESPAÇO E DESEMPENHO)
# ====================================================================

def gerar_pdfs_estatisticas():
    
    h_vals = np.linspace(0.01, 3.5, 300)
    ciclo_cores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Análise 1: Impacto do Fator Kappa
    print(" -> Plotando PDF: Variação de Kappa...")
    plt.figure(figsize=(10, 6))
    for i, k in enumerate([1.0, 5.0, 15.0]):
        sistema = Params(kappa=k, mu=1.5, md=5.0)
        cor = ciclo_cores[i % len(ciclo_cores)]
        
        gerador = GeradorInversaNumerica(funcao_pdf=sistema.pdf_fHx)
        amostras_h = gerador.gerar_amostras(NUM_AMOSTRAS_PDF)
        
        plt.hist(amostras_h, bins=150, density=True, alpha=0.4, color=cor)
        plt.plot(h_vals, [sistema.pdf_fHx(x) for x in h_vals], linewidth=2.5, color=cor, label=fr'$\kappa$ = {k}')
        
    plt.title(r'PDF do Canal: Componente Dominante ($\kappa$) ($\mu=1.5$, $m_d=5.0$)')
    plt.xlabel('Amplitude do Desvanecimento ($H$)')
    plt.ylabel('Densidade de Probabilidade $f_H(x)$')
    plt.xlim(0, 3.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/01_Estatistica_PDF_Kappa.png', dpi=300)
    plt.close()

    # Análise 2: Impacto do Sombreamento (md)
    print(" -> Plotando PDF: Variação de Sombreamento...")
    plt.figure(figsize=(10, 6))
    for i, md in enumerate([1.01, 3.0, 20.0]):
        sistema = Params(kappa=5.0, mu=1.5, md=md)
        cor = ciclo_cores[i % len(ciclo_cores)]
        
        gerador = GeradorInversaNumerica(funcao_pdf=sistema.pdf_fHx)
        amostras_h = gerador.gerar_amostras(NUM_AMOSTRAS_PDF)
        
        label_md = r'$\infty$' if md >= 20.0 else ( '1' if md <= 1.01 else md )
        prefixo = '->' if md >= 20.0 or md <= 1.01 else '='
        
        plt.hist(amostras_h, bins=150, density=True, alpha=0.4, color=cor)
        plt.plot(h_vals, [sistema.pdf_fHx(x) for x in h_vals], linewidth=2.5, color=cor, label=fr'$m_d$ {prefixo} {label_md}')
        
    plt.title(r'PDF do Canal: Sombreamento ($m_d$) ($\kappa=5.0$, $\mu=1.5$)')
    plt.xlabel('Amplitude do Desvanecimento ($H$)')
    plt.ylabel('Densidade de Probabilidade $f_H(x)$')
    plt.xlim(0, 3.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/02_Estatistica_PDF_md.png', dpi=300)
    plt.close()

def gerar_heatmap_espacial():
    
    num_usuarios = 5_000
    gamma_t_linear = 10 ** (100.0 / 10) 
    sistema = Params(kappa=5.0, mu=1.5, md=3.0, path_loss_expoent=2.4, r_m=20, antenna_height=10)
    
    raios_r = sistema.gerar_R(num_usuarios)
    angulos_theta = np.random.uniform(0, 2 * np.pi, num_usuarios)
    x_coords = raios_r * np.cos(angulos_theta)
    y_coords = raios_r * np.sin(angulos_theta)

    distancias_d_r = sistema.calcular_D_R()
    path_loss = sistema.rho_t(distancias_d_r)
    
    gerador = GeradorInversaNumerica(funcao_pdf=sistema.pdf_fHx)
    amostras_h = gerador.gerar_amostras(num_usuarios)
    
    snr_instante_linear = gamma_t_linear * path_loss * (amostras_h ** 2)
    snr_instante_db = 10 * np.log10(np.maximum(snr_instante_linear, 1e-15)) 

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(x_coords, y_coords, c=snr_instante_db, cmap='viridis', s=10, alpha=0.8)
    plt.plot(0, 0, 'r^', markersize=12, label='Antena Central')
    plt.colorbar(scatter, label='SNR Instantânea (dB)')
    plt.title(fr'Heatmap de Cobertura ($R_M$={sistema.r_m}m, $h_a$={sistema.antenna_height}m) ($\kappa=5.0$, $\mu=1.5$, $m_d=3.0$, $\alpha=2.4$)')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/03_Espacial_Heatmap.png', dpi=300)
    plt.close()

def gerar_dimensionamento_geometrico():
    
    cenarios = [(5.0, 15.0), (10.0, 30.0), (15.0, 20.0), (20.0, 50.0)]
    curvas_op, curvas_sep = [], []
    
    for ha, rm in cenarios:
        sistema = Params(kappa=5.0, mu=1.5, md=5.0, path_loss_expoent=2.4, antenna_height=ha, r_m=rm)
        res_op, res_sep, _ = simular_metricas(sistema, NUM_AMOSTRAS_OP_SEP, SNR_GRID, gamma_th=1.0)
        
        label = fr'$h_a$={ha}m, $R_M$={rm}m'
        curvas_op.append((sistema.calcular_OP_teorica(SNR_GRID, 1.0), f'Teórica ({label})'))
        curvas_op.append((res_op, f'Simulada ({label})'))
        
        curvas_sep.append((sistema.calcular_ASEP_teorica(SNR_GRID), f'Teórica ({label})'))
        curvas_sep.append((res_sep, f'Simulada ({label})'))

    plotar_curva(SNR_GRID, curvas_op, r'OP: Dimensionamento Geométrico ($\kappa=5.0$, $\mu=1.5$, $m_d=5.0$, $\alpha=2.4$)', 'Outage Probability ($P_o$)', 'log', arquivo='images/04_OP_Geometria.png', y_lim=(1e-6, 1.0))
    plotar_curva(SNR_GRID, curvas_sep, r'ASEP: Dimensionamento Geométrico ($\kappa=5.0$, $\mu=1.5$, $m_d=5.0$, $\alpha=2.4$)', 'Average Symbol Error Probability ($P_e$)', 'log', arquivo='images/05_SEP_Geometria.png', y_lim=(1e-6, 1.0))

def gerar_op_feixe_vs_sombreamento():
    
    cenarios = [(2.0, 1.5), (2.0, 10.0), (15.0, 1.5), (15.0, 10.0)]
    curvas = []
    
    for k, md in cenarios:
        sistema = Params(kappa=k, mu=1.5, md=md, path_loss_expoent=2.4, r_m=15.0)
        res_op, _, _ = simular_metricas(sistema, NUM_AMOSTRAS_OP_SEP, SNR_GRID, gamma_th=1.0)
        
        label = fr'$\kappa$={k}, $m_d$={md}'
        curvas.append((sistema.calcular_OP_teorica(SNR_GRID, 1.0), f'Teórica ({label})'))
        curvas.append((res_op, 'Simulada (Ponto)'))

    plotar_curva(SNR_GRID, curvas, r'OP: Componente Dominante vs Sombreamento ($\mu=1.5$, $\alpha=2.4$, $R_M=15.0$)', 'Outage Probability ($P_o$)', 'log', arquivo='images/06_OP_Kappa_md.png', y_lim=(1e-6, 1.0))

def gerar_sep_visada_vs_clusters():
    
    cenarios = [(1.0, 1.0), (1.0, 3.0), (5.0, 1.0), (5.0, 3.0), (10.0, 1.0), (10.0, 3.0), (20.0, 1.0), (20.0, 3.0)]
    curvas = []
    
    for k, mu in cenarios:
        sistema = Params(kappa=k, mu=mu, md=5.0, path_loss_expoent=2.4, r_m=15.0)
        _, res_sep, _ = simular_metricas(sistema, NUM_AMOSTRAS_OP_SEP, SNR_GRID)
        
        label = fr'$\kappa$={k}, $\mu$={mu}'
        curvas.append((sistema.calcular_ASEP_teorica(SNR_GRID), f'Teórica ({label})'))
        curvas.append((res_sep, 'Simulada (Ponto)'))
        
    plotar_curva(SNR_GRID, curvas, r'ASEP: Componente Dominante vs Clusters ($m_d=5.0$, $\alpha=2.4$, $R_M=15.0$)', 'Average Symbol Error Probability ($P_e$)', 'log', arquivo='images/07_SEP_Kappa_mu.png', y_lim=(1e-6, 1.0))

def gerar_ce_limite_banda():
    
    cenarios = [(1.5, 2.0), (10.0, 2.0), (1.5, 2.5), (10.0, 2.5)]
    plt.figure(figsize=(9, 6))
    ciclo_cores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for idx, (md, alpha) in enumerate(cenarios):
        sistema = Params(kappa=5.0, mu=1.5, md=md, path_loss_expoent=alpha, r_m=8.0)
        _, _, res_ce = simular_metricas(sistema, NUM_AMOSTRAS_CE, SNR_GRID_CE)
        
        label_md = r"\infty" if md == 10.0 else md
        label = fr'$m_d \to {label_md}$, $\alpha$={alpha}'
        
        cor = ciclo_cores[idx % len(ciclo_cores)]
        plt.plot(SNR_GRID_CE, res_ce, linewidth=2.5, marker='+', markersize=6, color=cor, label=f'CE ({label})')

    plt.title(r'Capacidade Ergódica: O Impacto do Sombreamento e Atenuação ($\kappa=5.0$, $\mu=1.5$, $R_M=8.0$)')
    plt.xlabel(r'Transmit SNR, $\gamma_t$ (dB)')
    plt.ylabel('Ergodic Capacity (bits/s/Hz)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/08_CE_Banda.png', dpi=300)
    plt.close()

# ====================================================================
# EXECUÇÃO 
# ====================================================================
def main():
    preparar_diretorios()

    gerar_pdfs_estatisticas()
    gerar_heatmap_espacial()
    gerar_dimensionamento_geometrico()
    gerar_op_feixe_vs_sombreamento()
    gerar_sep_visada_vs_clusters()
    gerar_ce_limite_banda()
    
if __name__ == "__main__":
    main()