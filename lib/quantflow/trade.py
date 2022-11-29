import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import numpy as np
import math


# 交易信号计算
class TradeMode:
    def __init__(self, rf=0.002, commission=0.005, mode='absolute'):
        self.rf = rf
        self.commission = commission
        self.mode = mode
        self.signal = None

    @tf.function
    def cal_profit(self, x, y):
        # 0 hold; 1 buy; 2 sold x: [None, windows_width] y: [None, label_width]
        modified_input1 = x[:, -1] * (1 - self.rf - self.commission)
        modified_input2 = x[:, -1] * (1 + self.rf + self.commission)
        if self.mode == 'absolute':
            self.signal = tf.where(y[:, -1] < modified_input1, 2, 0) + tf.where(
                y[:, -1] > modified_input2, 1, 0)
        elif self.mode == 'diff':
            self.signal = y[:, -1] - x[:, -1]
        elif self.mode == 'mean':
            self.signal = tf.where(tf.math.reduce_mean(y, axis=-1) < modified_input1, 2, 0) + tf.where(
                tf.math.reduce_mean(y, axis=-1) > modified_input2, 1, 0)
        return self.signal

    def plot(self, x, y, signals, marker_size=5):
        # plot label curve
        scatter1 = None
        scatter2 = None
        scatter3 = None
        plt.plot(x, y, color='black', linestyle='dashed', linewidth=1)
        if self.mode != 'diff':
            # plot dots
            if signals[0] == 0:
                scatter1, = plt.plot(x[-1], y[-1], color='black', marker='o', linestyle='dashed', linewidth=1,
                                    markersize=1 * marker_size)
            elif signals[0] == 1:
                scatter2, = plt.plot(x[-1], y[-1], color='red', marker='o', linestyle='dashed', linewidth=1,
                                     markersize=1 * marker_size)
            else:
                scatter3, = plt.plot(x[-1], y[-1], color='green', marker='o', linestyle='dashed', linewidth=2,
                                     markersize=1 * marker_size)
            if scatter2 is None:
                scatter2, = plt.plot([0], [0], color='red', marker='o', linestyle='dashed', linewidth=1,
                                    markersize=1 * marker_size)
            if scatter3 is None:
                scatter3, = plt.plot([0], [0], color='green', marker='o', linestyle='dashed', linewidth=2,
                                    markersize=1 * marker_size)
            if scatter1 is None:
                scatter1, = plt.plot([0], [0], color='black', marker='o', linestyle='dashed', linewidth=1,
                                    markersize=1 * marker_size)
            plt.legend(handles=[scatter1, scatter2, scatter3],
                       labels=["hold", "buy", "sell"], loc='best')


# 回调
def quant_metrics(hp, hm, period=24, rf=0.04):
    pp = np.array([])
    pm = np.array([])
    # 计算区间收益
    for i in range(0, len(hp) - 24, 24):
        pp = np.append(pp, hp[i:i + 24][-1] - hp[i:i + 24][0])
        pm = np.append(pm, hm[i:i + 24][-1] - hm[i:i + 24][0])
    # 计算最小单位收益
    hpr = [(hp[i + 1] - hp[i]) / hp[i] for i in range(len(hp) - 1)]
    hmr = [(hm[i + 1] - hm[i]) / hp[i] for i in range(len(hm) - 1)]
    ppr = []
    pmr = []
    # 计算区间累计收益率
    for i in range(0, len(hp) - 24, 24):
        ppr.append((hp[i:i + 24][-1] - hp[i:i + 24][0]) / hp[i:i + 24][0])
        pmr.append((hm[i:i + 24][-1] - hm[i:i + 24][0]) / hm[i:i + 24][0])
    # 基准年化收益率 Baseline Annualized rate of return
    annual_mi = (pow((hm[-1] - hm[0]) / hm[0] + 1, 8760 / len(hm)) - 1)
    print("基准平均区间收益率 Baseline Interval rate of return(" + str(period) + " h): ", np.mean(pmr) * 100, " %")
    print("基准年化收益率 Baseline Annualized rate of return(yr/h)", annual_mi * 100, " %")
    print('基准回测收益率(%s h): %s' % (len(hm), (hm[-1] - hm[0]) * 100 / hp[0]), "%")
    print("-----------------------------------------------------------------------")
    # 策略年化收益率  Annualized rate of return 
    annual_ret = (pow((hp[-1] - hp[0]) / hp[0] + 1, 8760 / len(hp)) - 1)
    print("策略平均区间收益率 Interval rate of return(" + str(period) + " h): ", np.mean(ppr) * 100, " %")
    print("策略年化收益率  Annualized rate of return (yr/h)", annual_ret * 100, " %")
    print('策略回测收益率  backtesting rate of return(%s h): %s' % (len(hp), (hp[-1] - hp[0]) * 100 / hp[0]), "%")
    print("-----------------------------------------------------------------------")
    # 其他指标
    beta = (np.cov(pp, pm)[0][1] / np.var(pm))
    print("超额收益率 abnormal return: %.4f" % ((hp[-1] - hm[-1]) * 100 / hp[0]), " %")
    print("年化夏普 Sharpe(区间 interval): %.3f" % (
                ((np.array(ppr) - 0.04 / 360).mean() * math.sqrt(360)) / (np.array(ppr) - 0.04 / 360).std()))
    print("策略最大回撤 Maximum Drawdown(h): ", ((np.maximum.accumulate(hp) - hp) / np.maximum.accumulate(hp)).max() * 100, " %")
    print("策略最大回撤 Maximum Drawdown(" + str(period) + " h): ",
          ((np.maximum.accumulate(pp + hp[0]) - (pp + hp[0])) / np.maximum.accumulate(pp + hp[0])).max() * 100, " %")
    print("beta(年化公式法 annualized  /%sh): " % period, beta)
    print("alpha(年化公式法 annualized  /%sh): " % period, np.mean(annual_ret - (rf + beta * (annual_mi - rf))))
    print("-----------------------------------------------------------------------")
    print("**yr/h****")
    beta, alpha, r_value, p_value, std_err = stats.linregress(hmr, hpr)
    print("beta(回归法 /24h): %s\nalpha(回归法): %s\nr_value(回归法): %s\np_value(回归法): %s\n标准误差(回归法): %s\n" % (beta, alpha * 8760, r_value, p_value, std_err))
    print("-----------------------------------------------------------------------")
    print("**yr/day****")
    beta, alpha, r_value, p_value, std_err = stats.linregress(pmr, ppr)
    print("beta(回归法 /24h): %s\nalpha(回归法): %s\nr_value(回归法): %s\np_value(回归法): %s\n标准误差(回归法): %s\n" % (beta, alpha * 365, r_value, p_value, std_err))


def trade_callback(y_pred, y_actual, fund=10000, hedge_rate=0.2, rf=.002, mode='rate'):
    # 简单反向交易对冲回调
    trade_count = 0
    current_fund = fund
    target_quant = 0
    start_quant = fund / y_actual[0]
    hp = []
    hm = []
    text = ""
    for index, price in enumerate(y_actual[:-1]):
        value = current_fund + target_quant * price * (1 - rf)
        if y_pred[index] == 1:
            trade_count = trade_count + 1
            if mode == 'rate':
                target_quant = target_quant + (current_fund * hedge_rate * (1 - rf)) / price
                current_fund = current_fund - current_fund * hedge_rate
            elif mode == 'absolute':
                if value * hedge_rate <= current_fund:
                    target_quant = target_quant + (value / price) * hedge_rate * (1 - rf)
                    current_fund = current_fund - value * hedge_rate
                else:
                    target_quant = target_quant + (current_fund / price) * (1 - rf)
                    current_fund = 0
        elif y_pred[index] == 0:
            if mode == 'rate':
                trade_count = trade_count + 1
                current_fund = current_fund + target_quant * hedge_rate * price * (1 - rf)
                target_quant = target_quant - target_quant * hedge_rate
            elif mode == 'absolute':
                if value * hedge_rate * (1 - rf) / price <= target_quant:
                    target_quant = target_quant - value * hedge_rate * (1 - rf) / price
                    current_fund = current_fund + value * hedge_rate
                else:
                    current_fund = current_fund + target_quant * price * (1 - rf)
                    target_quant = 0
        hp.append(current_fund + target_quant * price * (1 - rf))
        hm.append(start_quant * price * (1 - rf))
        text = text + "current fund: " + str(current_fund) + "   target quant  " + str(
            target_quant) + "   target price: " + str(price) + "   value: " + str(
            current_fund + target_quant * price * (1 - rf)) + "\n"
        print("current fund: ", current_fund, "   target quant  ", target_quant,
              "   target price: ", price, "   value: ", (current_fund + target_quant * price))
    price = y_actual[-1]
    current_fund = current_fund + target_quant * price * (1 - rf)
    target_quant = 0
    hp.append(current_fund)
    hm.append(start_quant * price * (1 - rf))
    text = text + "current fund: " + str(current_fund) + "   target quant  " + str(
        target_quant) + "   target price: " + str(price) + "   value: " + str(current_fund) + "\n"
    print("current fund: ", current_fund, "   target quant  ", target_quant, "   target price: ",
          price, "   value: ", (current_fund + target_quant * price))
    return int(current_fund + target_quant * y_actual[-1]), trade_count, len(y_pred), text, hp, hm
