import numpy as np
from scipy.special import gamma, kv, gammainc, gammaln, hyp2f1
import mpmath
from scipy.integrate import quad

ANTENA_HEIGHTS = 3
USER_HEIGHTS = 1.5
R_M = 100
FREQUENCY = 10 * 10e9
LAPLACE = 0
LIGHT_SPEED = 3 * 10e8


class Params:
    def __init__(self, kappa=2.0, mu=1.0, md=3.0, r_m=R_M, antenna_height=ANTENA_HEIGHTS, user_height=USER_HEIGHTS, path_loss_expoent=2.0, d_0=1.0):
        self.kappa = kappa
        self.mu = mu
        self.md = md
        self.K_Rice = 0
        self.Pt = 1.0
        self.lambda_0 = LIGHT_SPEED / FREQUENCY
        self.sigma_e2_n = 1e-9
        self.close_space_reference = 1

        self.path_loss_expoent = path_loss_expoent 
        self.r_m = r_m
        self.antenna_height = antenna_height
        self.user_height = user_height
        self.d_0 = d_0
        
        self.R_vals = None   
        self.D_R_vals = None
        
    def recieved_signal(self, Pt, rho_t, H, S, N):
        aux = np.sqrt(Pt * rho_t)
        Y = aux * H * S + N
        return Y
    
    def rho_t(self, D_R):
        aux_1 = (self.lambda_0 / (4 * np.pi * self.close_space_reference)) ** 2
        aux_2 = (D_R / self.close_space_reference) ** -self.path_loss_expoent
        return aux_1 * aux_2

    def gerar_R(self, num_users: int) -> np.ndarray:
        U = np.random.rand(num_users)
        self.R_vals = self.r_m * np.sqrt(U)
        return self.R_vals
    
    def calcular_D_R(self) -> np.ndarray:

        if self.R_vals is None:
            raise ValueError("Não há distâncias radiais R geradas.")
        self.D_R_vals = np.sqrt((self.user_height - self.antenna_height)**2 + self.R_vals**2)
        return self.D_R_vals

    def _log_coeficiente_Ad(self, n: int) -> float:

        k = self.kappa   
        u = self.mu 
        m = self.md 

        arg_bessel = 2 * np.sqrt((m - 1) * k * u) 
        ordem_bessel = abs(m - n)

        termo1 = ((m + n) / 2) * np.log((m - 1) * k)
        termo2 = (n + u) * np.log(1 + k)
        termo3 = (((3 * n + m) / 2) + u) * np.log(u)
        
        termo_log = termo1 + termo2 + termo3 - gammaln(n + 1) - gammaln(m) - gammaln(n + u)
        bessel = kv(ordem_bessel, arg_bessel)
        return np.log(2.0) + termo_log + np.log(np.maximum(bessel, np.finfo(float).tiny))

    def _coeficiente_Ad(self, n: int) -> float:
        return np.exp(self._log_coeficiente_Ad(n))
    
    def pdf_fHx(self, x: float, max_terms: int = 150) -> float:
        k = self.kappa
        u = self.mu
        pdf_valor = 0.0
        expoente = np.minimum((1 + k) * u * (x**2), 700)
        denominator = np.exp(expoente)
        
        peak_n = int(k * u)

        for n in range(max_terms):
            log_Ad = self._log_coeficiente_Ad(n)
            numerator_x = x ** (2 * n + 2 * u - 1)

            log_actual = np.log(2.0) + log_Ad + np.log(np.maximum(numerator_x, np.finfo(float).tiny)) - np.log(denominator)
            actual = np.exp(log_actual)
            pdf_valor += actual

            if actual < 1e-12 and n > peak_n + 10:
                break
        return pdf_valor

    def SNR(self, amostras_H: np.ndarray, D_R) -> np.ndarray:
        gamma_t = (self.Pt / self.sigma_e2_n)
        path_loss = self.rho_t(D_R)
        ganho_potencia_canal = amostras_H ** 2
        Gamma = gamma_t * path_loss * ganho_potencia_canal
        return Gamma

    def calcular_gamma_0(self, gamma_t_linear):
        aux = (self.lambda_0 / (4 * np.pi)) ** 2
        aux = aux * (self.d_0 ** (self.path_loss_expoent - 2))
        return gamma_t_linear * aux
     
    def calcular_CL(self, gamma_0):
        L = np.abs(self.antenna_height - self.user_height)
        CL = (1 / gamma_0) * (1 + self.kappa) * self.mu * (L ** self.path_loss_expoent)
        return CL
    
    def calcular_CU(self, gamma_0):
        L = np.abs(self.antenna_height - self.user_height)
        CU = (1 / gamma_0) * (1 + self.kappa) * self.mu * ((L ** 2 + self.r_m ** 2) ** (self.path_loss_expoent / 2))
        return CU    

    def gamma_inc_inferior(self, s, x): 
        return gammainc(s, x) * gamma(s)

    def calcular_Q(self, n, a, b, c):
        
        if np.isscalar(c):
            if c < 1e-14: return 0.0
        else:
            c = np.where(c < 1e-14, 1e-14, c)
            
        termo1 = self.gamma_inc_inferior(n + a, c)
        termo2 = self.gamma_inc_inferior(n + a + (2.0 / b), c) / (c ** (2.0 / b))
        return termo1 - termo2

    def theorical_Fgamma(self, gamma_0, n_max, gamma_th):
        
        coeficiente_1 = (1 / (self.r_m ** 2)) * (gamma_0 ** (2 / self.path_loss_expoent))

        CL = self.calcular_CL(gamma_0)
        CU = self.calcular_CU(gamma_0)

        soma = np.zeros_like(gamma_0, dtype=float)
        
        for n in range(n_max):
            log_Ad = self._log_coeficiente_Ad(n)
            log_den = (n + self.mu + (2 / self.path_loss_expoent)) * np.log((1 + self.kappa) * self.mu)
            termo_1 = np.exp(log_Ad - log_den)

            CU_comexpoente = CU ** (2.0 / self.path_loss_expoent)
            CL_comexpoente = CL ** (2.0 / self.path_loss_expoent)
            
            Q_CU = self.calcular_Q(n, self.mu, self.path_loss_expoent, CU * gamma_th)
            Q_CL = self.calcular_Q(n, self.mu, self.path_loss_expoent, CL * gamma_th)
            
            termo_interno = (CU_comexpoente * Q_CU - CL_comexpoente * Q_CL)
            soma += termo_1 * termo_interno
        
        F_Gamma = coeficiente_1 * soma
        return F_Gamma

    def calcular_OP_teorica(self, snr_db_grid, gamma_th=1.0, n_max=100):

        gamma_t_linear = 10 ** (snr_db_grid / 10)
        gamma_0_array = self.calcular_gamma_0(gamma_t_linear)
        
        op_teorica = self.theorical_Fgamma(gamma_0=gamma_0_array, n_max=n_max, gamma_th=gamma_th)
        return op_teorica
    
    def calcular_P(self, b_val, n, x):
        
        delta = self.path_loss_expoent
        u = self.mu

        z = x / (x + b_val / 2.0)
        f1 = hyp2f1(1, n + u + 0.5, n + u + 1, z)
        f2 = hyp2f1(1, n + u + 0.5, n + u + (2.0 / delta) + 1, z)
        
        log_gamma = gammaln(n + u + 0.5)
        log_x = (n + u) * np.log(x)

        log_den1 = np.log(n + u) + (n + u + 0.5) * np.log(x + b_val / 2.0)
        log_den2 = np.log(n + u + 2.0 / delta) + (n + u + 0.5) * np.log(x + b_val / 2.0)

        log_t1 = log_gamma + log_x + np.log(np.maximum(np.abs(f1), np.finfo(float).tiny)) - log_den1
        log_t2 = log_gamma + log_x + np.log(np.maximum(np.abs(f2), np.finfo(float).tiny)) - log_den2

        sign_t1 = np.where(np.isfinite(f1) & (f1 != 0), np.sign(f1), 1.0)
        sign_t2 = np.where(np.isfinite(f2) & (f2 != 0), np.sign(f2), 1.0)

        t1 = sign_t1 * np.exp(log_t1)
        t2 = sign_t2 * np.exp(log_t2)

        return t1 - t2

    def calcular_ASEP_teorica(self, snr_db_grid, a=1.0, b=2.0, n_max=100):
        
        gamma_t_linear = 10 ** (snr_db_grid / 10)
        gamma_0 = self.calcular_gamma_0(gamma_t_linear)
        delta = self.path_loss_expoent
        
        CL = self.calcular_CL(gamma_0)
        CU = self.calcular_CU(gamma_0)
        
        constante_ext = (a * np.sqrt(b)) / (2 * np.sqrt(2 * np.pi) * (self.r_m ** 2))
        coef_gamma0 = gamma_0 ** (2.0 / delta)
        
        soma = np.zeros_like(gamma_0, dtype=float)
        
        for n in range(n_max):
            log_Ad = self._log_coeficiente_Ad(n)
            log_den = (n + self.mu + 2.0 / delta) * np.log((1 + self.kappa) * self.mu)
            termo_1 = np.exp(log_Ad - log_den)
            
            P_CU = self.calcular_P(b, n, CU)
            P_CL = self.calcular_P(b, n, CL)
            
            termo_interno = (CU ** (2.0 / delta)) * P_CU - (CL ** (2.0 / delta)) * P_CL
            soma += termo_1 * termo_interno
            
        return constante_ext * coef_gamma0 * soma

    def _meijer_g_capacity(self, n, z_array):
        
        delta = self.path_loss_expoent
        u = self.mu
        
        an = [-2.0/delta, -1.0]
        ap = [0.0]
        bm = [n + u - 1.0, -1.0, -1.0]
        bq = [-2.0/delta - 1.0]
        
        res = np.zeros_like(z_array, dtype=float)
        for i, z in enumerate(z_array):
            res[i] = float(mpmath.meijerg([an, ap], [bm, bq], float(z)))
        return res

    def calcular_CE_teorica(self, snr_db_grid, n_max=60):
        
        gamma_t_linear = 10 ** (snr_db_grid / 10)
        gamma_0 = self.calcular_gamma_0(gamma_t_linear)
        delta = self.path_loss_expoent
        
        CL = self.calcular_CL(gamma_0)
        CU = self.calcular_CU(gamma_0)
        
        constante_ext = (1.0 / np.log(2)) * (2.0 / (delta * (self.r_m ** 2)))
        coef_gamma0 = gamma_0 ** (2.0 / delta)
        
        soma = np.zeros_like(gamma_0, dtype=float)
        
        for n in range(n_max):
            log_Ad = self._log_coeficiente_Ad(n)
            log_den = (n + self.mu + 2.0 / delta) * np.log((1 + self.kappa) * self.mu)
            termo_1 = np.exp(log_Ad - log_den)
            
            G_CU = self._meijer_g_capacity(n, CU)
            G_CL = self._meijer_g_capacity(n, CL)
            
            
            termo_interno = (CU ** ((2.0 / delta) + 1.0)) * G_CU - (CL ** ((2.0 / delta) + 1.0)) * G_CL
            soma += termo_1 * termo_interno
            
        return constante_ext * coef_gamma0 * soma
        
    def calcular_CE_integracao(self, snr_db_grid, n_max=60):
        
        gamma_t_linear = 10 ** (snr_db_grid / 10)
        gamma_0_array = self.calcular_gamma_0(gamma_t_linear)
        
        ce_teorica = np.zeros_like(gamma_0_array, dtype=float)
        
        for i, gamma_0_val in enumerate(gamma_0_array):
            def integrando(gamma):
                cdf_val = self.theorical_Fgamma(np.array([gamma_0_val]), n_max=n_max, gamma_th=gamma)[0]
                return (1.0 - np.clip(cdf_val, 0.0, 1.0)) / (1.0 + gamma)
            
            resultado_integral, _ = quad(integrando, 0, np.inf, limit=200)
            ce_teorica[i] = resultado_integral / np.log(2)
            
        return ce_teorica