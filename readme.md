Ideas came from: https://www.quantifiedstrategies.com/quantitative-trading-strategies/
  
The code provides visualization results

Project is continueing

# Russel (2000) rebalancing strategy:
## Basic Knowledge:
罗素3000指数（指数代号：RUA）是一个资本加权的股票市场指数，旨在于成为整个美国股票市场的基准。它包括了在美国注册的总市值最大的3,000家上市公司，约占美国公共股票市场的 97%.

罗素2000指数（英语：Russell 2000 Index）为罗素3000指数中收录市值最小的2000家（排序第1001–3000名）公司股票进行编制。罗素2000指数市值只占罗素3000指数约10%，但因被视为最能代表美国市场小型股的重要指数，备受市场和媒体关注.

## Logic:
这是一种利用罗素 2000 指数在 6 月底年度调整期间异常优异表现的方法。交易者在 6 月 23 日后的第一个交易日收盘时买入，在 7 月的第一个交易日收盘时卖出，利用这一时期通常出现的反弹。

罗素 2000 指数通常每年进行一次再平衡。具体日期可能会有所不同，但通常在 6 月份进行。在此期间，该指数会进行调整，以确保其准确代表美国股票市场的小盘股板块。

重新平衡包括增加符合指数标准的新公司，剔除不再符合标准的其他公司，以及调整现有成分股的权重。这一过程有助于保持指数的完整性和对投资者的相关性。

每年 6 月的第四个星期五，罗素 1000、罗素 2000、罗素 3000 和其他罗素指数都会重组。富时罗素会提前通知投资者他们应该期待的变动。

罗素 2000 指数每年在 6 月的第四个星期五对其持有的股票进行一次再平衡。显然，第四个星期五的日期各不相同，因此建议交易者查看罗素网站，了解具体时间。

罗素 2000 指数的再平衡是指调整罗素 2000 指数的成分股，以根据预定的标准保持小盘股的代表性。





# Rubber band trading strategy:
## Basic Knowledge:
橡皮筋策略符合逆向思维。它遵循沃伦-巴菲特的原则：在别人贪婪时恐惧，在别人恐惧时贪婪。当市场出现过度负面情绪时，就会出现买入机会；当市场出现亢奋情绪时，就会出现卖出机会。

当价格交易高于布林带上轨时，预示着市场超买，预计可能向均值反转。相反，当价格低于下限时，表明市场超卖，有可能向上反转。

橡皮筋交易策略是一种逆向/均值回复的股票交易方法。它旨在识别市场超买或超卖的点位，预期市场会向均值回弹。交易者可以使用凯尔特纳通道或布林线等指标来实施这一策略。

## Logic:
1. 计算（最高价 - 最低价）的 5 天平均值,这可以告诉你股票在 5 天内的平均涨跌幅度, 它被称为 ATR
2. 找出最近 5 天内的最高价, 这将为您提供近期 "该股近期涨幅 "的参考
3. 在 5 天高点下方创建一个 “波段” 取 5 天高点，然后减去 2.5 × ATR. 这定义了价格必须下跌到什么程度才算 “超卖”。
4. 如果收盘价低于该价位带，则买入, 如果当天收盘价低于该 “下限价位带”，则在收盘时买入，期待价格很快反弹.
5. 当价格收盘高于昨日高点时卖出, 你继续持有头寸，直到价格收盘高于昨日高点--这预示着反弹可能已经发生。

简单总结：
如果价格突然跌至远低于近期高点（基于波动率），则买入 - 当价格反弹至略高于昨日高点时卖出。



# MFI Indicator strategy:
## Basic Knowledge:
货币流通指数 (MFI) 是一种技术分析指标，它可以让交易者 “跟着资金走”。也就是说，该指标衡量的是特定时期内资金进出证券的情况。通过观察 MFI，交易者可以判断相关资产是否存在买入或卖出压力, MFI 有助于发出超买或超卖信号

MFI 值将在 0 和 100 之间波动, MFI 读数高于 80 通常意味着市场处于超买状态，而低于 20 则意味着市场处于超卖状态

MFI 读数超过 80 意味着超买条件。当价格处于超买状态时，应寻找机会在市场上发出卖单，以期待趋势反转。读数低于 20 意味着市场处于超卖状态。当价格处于超卖状态时，应寻找机会在市场上下达买单，以期待趋势反转

## Logic:
MFI 计算
1. 确定相关时间段的典型价格 (TP), TP = (HIGH + LOW + CLOSE)/3
2. 资金流 (MF), MF = TP * VOLUME
3. 确定货币比率 (MR), MR = 正货币流 (PMF)/ 负货币流 (NMF)
4. 计算货币流指数 (MFI), MFI = 100 - (100 / (1 + MR))



# S&P 500, gold, and bonds rotation momentum strategy:
## Basic Knowledge:
标准普尔500指数（英语：Standard & Poor's 500），简称S&P 500 、标普500或史坦普500，自1957年起记录美国股市表现，涵盖500只普通股，是最受关注的股票指数之一。该指数覆盖了美国约80%的公开上市公司总市值

指数内的500只普通股（包括不动产投资信托）都是在美国股市的两大股票交易市场，纽约证券交易所和美国全国证券业协会行情自动传报系统（纳斯达克、NASDAQ）中有多个交易的公司。几乎所有标准普尔中的公司都是全美最高金额买卖的500只股票。这个股票指数由标准普尔公司建立并维护

动量策略是一种众所周知的策略，几十年来一直表现出色。它在长时间段内不起作用，但在 1 至 12 个月的半长时间段内效果最佳

例如，有一种表现良好的策略是购买过去 6 个月中表现最好的股票。每个月底，你对最佳股票进行排名，然后买入最佳的 x 只股票，并持有一个月。下个月结束时，再次进行排名。以后每个月都这样做。

标准普尔 500 指数和国债作为一种基于动量和轮动的战术性资产配置策略，经常被提及。这是为什么呢？ 最有可能的解释是, 国债通常是避风港。当未来不确定时，许多投资者会寻求更多地配置国债等资产

## Logic:
著名货币经理人梅布-法贝尔（Meb Faber）在 2015 年发表了一篇文章，他在黄金、股票和债券之间进行了轮换。这是一种动量策略，交易规则很简单。

下面就是它的全部内容：

三类资产： 股票、债券、黄金。
stocks (S&P 500), bonds, and gold

无论哪种资产在上涨（定义为 3 个月均线大于 10 个月均线），都平均投资。

我们假设资金分配是均等的，这取决于我们得到多少信号。例如，如果一个月内有两个积极信号，我们就分配 50%给每个仓位；如果有三个信号，我们就分配 33.33%给每个仓位；如果只有一个信号，我们就分配 100%。



# The turn of the month strategy:
## Basic Knowledge:
研究表明，股票几乎在当月的最后五个交易日和新的一个月的前三个交易日全部上涨。
我们在当月倒数第五个交易日收盘时做多，七天后退出，即在下个月第三个交易日收盘时退出。
## Logic:
月末效应（又称 Ultimo 效应）是股市中一种著名的交易策略。我们曾经介绍过这种反常现象。也许鲜为人知的是，这种效应会蔓延到下个月的前三天。因此，更正确的名称可能是月轮策略（或月轮效应）。

月度转折策略在大多数市场中都表现出色，尽管暴露时间较短，但经常战胜买入并持有策略。我们在当月倒数第五个交易日收盘时做多，七天后退出，即在下个月第三个交易日收盘时退出。



# Stock prediction using LSTM (time series analysis)
## concept
LSTM (Long Short-Term Memory) networks are particularly well-suited for stock price prediction because they are designed to capture long-term dependencies and patterns in sequential data. Unlike traditional models, LSTM can remember information over many time steps and selectively retain or forget past data using its gating mechanism, making it effective in identifying trends, cycles, and nonlinear relationships in stock prices. This ability to model complex temporal dynamics allows LSTM to learn from historical price movements and other market indicators, even in the presence of noise and volatility, which are common in financial data.

# GAN for Stock Prediction
idea: https://github.com/hungchun-lin/Stock-price-prediction-using-GAN.git
using GAN to predict the future stock, the generator is the LSTM.
