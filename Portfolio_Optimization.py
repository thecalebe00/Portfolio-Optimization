#Importação de bibliotecas
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
!pip install yfinance #biblioteca para importação dos dados das ações
import yfinance as yf
!pip install pyportfolioopt #biblioteca a ser utilzada para otimização da carteira

#Seleção de ativos
ativos = input("Digite quais são os ativos que vão compor sua carteira: ").upper().split(",")
ativos = [ticker + ".SA" for ticker in ativos]

#Com as devidas bibliotecas instaladas, agora iremos fazer a composição da carteira pelas ações escolhidas
carteira = pd.DataFrame()
for i in ativos:
  carteira[i] = yf.download(i,period="3y")['Adj Close']

#Visualização dos retornos das ações
plt.style.use('dark_background')
carteira.plot(figsize=(10,5))

#Para a realização da fronteira eficiente, é necessário a realização do retorno esperado e da matriz de covariação
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
er = mean_historical_return(carteira) #Retorno esperado da carteira
sd = CovarianceShrinkage(carteira).ledoit_wolf() #Matriz de covariância
sd

#Com a matriz de covariação e o retorno esperado das ações da carteira, agora calculamos a porcentagemd e alocação em cada posição da carteira
taxa_livre_de_risco = float(input("Qual a taxa livre de risco? *Digite usando o separador decimal usando o ponto* "))
taxa_livre_de_risco = taxa_livre_de_risco/100
from pypfopt.efficient_frontier import EfficientFrontier
ef = EfficientFrontier(er,sd)
composicao = ef.max_sharpe(risk_free_rate=taxa_livre_de_risco)

#Visão geral
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.txt")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#Como houveram posições com peso igual a 0, houve o rebalanceamento de modo a ponderar melhor as posições da carteira
from pypfopt import objective_functions

ef = EfficientFrontier(er, sd)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
w = ef.max_sharpe()
ef.clean_weights()

#Quantidade de ações a serem compradas de acordo com o valor de seu portfolio
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
portfolio_value = float(input("Qual o valor do seu portfolio? * Digite apenas o numero, sem vírgulas e pontos *"))
latest_prices = get_latest_prices(carteira)
da = DiscreteAllocation(w, latest_prices, portfolio_value)
allocation, leftover = da.lp_portfolio()
print(allocation)

#Plotagem da Fronteira eficiente
from pypfopt import plotting
ef = EfficientFrontier(er, sd, weight_bounds=(None, None))
ef.add_constraint(lambda w: w[0] >= 0.2)
ef.add_constraint(lambda w: w[2] == 0.15)
ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.show()

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Geração de portfolios aleatórios, para a composição da fronteira
n_samples = 10000
w = np.random.dirichlet(np.ones(len(er)), n_samples)
rets = w.dot(er)
stds = np.sqrt(np.diag(w @ sd @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Resultado
ax.set_title("Fronteira eficiente com portfolios aleatórios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

#instalação da biblioteca de backtesting
!pip install bt
import bt
#Definição de estratégias e escolhas de argumentos/algoritmos para cada estratégia
rebalancing = bt.Strategy("Rebalanceamento",
                          [bt.algos.RunMonthly(run_on_end_of_period=True), 
                           bt.algos.SelectAll(), 
                           bt.algos.WeighSpecified(**composicao), 
                           bt.algos.Rebalance()]
                          )

buyhold = bt.Strategy("Buy & Hold",
                      [bt.algos.RunOnce(), 
                       bt.algos.SelectAll(), 
                       bt.algos.WeighSpecified(**composicao),
                       bt.algos.Rebalance()]
                      )

#Estratégias de investimento como variáveis, e com escolha de argumentos
bt1 = bt.Backtest(rebalancing, carteira, initial_capital=portfolio_value)
bt2 = bt.Backtest(buyhold, carteira, initial_capital=portfolio_value)

results = bt.run(bt1, bt2)

#Resumo dos resultados das estratégias de investimento
results.display()

results.plot(title="Comparações de estratégias",figsize=(13,5))

#Todas transações realizadas desde a rodagem do algoritmo
results.get_transactions()

#Distribuições de pesos nos ativos do portfólio
results.get_security_weights()

results.plot_security_weights()

#Correlação das estratégias de investimento
results.plot_correlation()

#Histograma dos retornos
results.plot_histogram()
