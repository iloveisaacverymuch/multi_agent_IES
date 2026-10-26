# CDA交易框架
def CDA(trade_re, P_re, trade_com, P_com, trade_in, P_in):
    #创造交易簿
    residential = [1, trade_re, P_re]
    commercial = [2, trade_com, P_com]
    industry = [3, trade_in, P_in]

    sell_e = [] #卖方交易簿
    buy_e = [] #买方交易簿
    if trade_re >= 0:
        sell_e.append(residential)
    else:
        buy_e.append(residential)

    if trade_com >= 0:
        sell_e.append(commercial)
    else:
        buy_e.append(commercial)

    if trade_in >= 0:
        sell_e.append(industry)
    else:
        buy_e.append(industry)

    for i in sell_e:
        i[1] = abs(i[1])
    for i in buy_e:
        i[1] = abs(i[1])

    # 卖方价格从低到高
    sell_sorted = sorted(sell_e, key=lambda x: x[2], reverse=False)
    # 买方出价从高到低
    buy_sorted = sorted(buy_e, key=lambda x: x[2], reverse=True)
    # 保存交易记录
    trading_record = []

    if len(sell_sorted) == 0 or len(buy_sorted) == 0:
        return sell_sorted, buy_sorted, []

    s = b = 1
    while buy_sorted[b - 1][2] >= sell_sorted[s - 1][2]:
        trade_amount = min(buy_sorted[b - 1][1], sell_sorted[s - 1][1])
        trade_price = (buy_sorted[b - 1][2] + sell_sorted[s - 1][2]) / 2
        buy_sorted[b - 1][1] = buy_sorted[b - 1][1] - trade_amount
        trading_record.append([buy_sorted[b - 1][0], sell_sorted[s - 1][0], trade_amount, trade_price])
        if buy_sorted[b - 1][1] == 0:
            b = b + 1
        sell_sorted[s - 1][1] = sell_sorted[s - 1][1] - trade_amount
        if sell_sorted[s - 1][1] == 0:
            s = s + 1
        if b > len(buy_sorted) or s > len(sell_sorted):
            break

    return sell_sorted, buy_sorted, trading_record

#CDA市场获利
def carbon_aware_CDA(
    trade_re,
    P_re,
    carbon_re,
    trade_com,
    P_com,
    carbon_com,
    trade_in,
    P_in,
    carbon_in,
    carbon_price=0.05,
):
    participants = [
        [1, trade_re, P_re, carbon_re],
        [2, trade_com, P_com, carbon_com],
        [3, trade_in, P_in, carbon_in],
    ]

    sell_e = []
    buy_e = []
    for original in participants:
        participant = original.copy()
        participant[1] = abs(participant[1])
        effective_price = participant[2] + carbon_price * max(participant[3], 0.0)
        participant.append(effective_price)
        if participant[1] <= 1e-9:
            continue
        if original[1] >= 0:
            sell_e.append(participant)
        else:
            buy_e.append(participant)

    sell_sorted = sorted(sell_e, key=lambda x: (x[4], x[2]), reverse=False)
    buy_sorted = sorted(buy_e, key=lambda x: x[2], reverse=True)
    trading_record = []

    if len(sell_sorted) == 0 or len(buy_sorted) == 0:
        return sell_sorted, buy_sorted, []

    s = b = 1
    while buy_sorted[b - 1][2] >= sell_sorted[s - 1][4]:
        trade_amount = min(buy_sorted[b - 1][1], sell_sorted[s - 1][1])
        trade_price = (buy_sorted[b - 1][2] + sell_sorted[s - 1][2]) / 2
        buy_sorted[b - 1][1] = buy_sorted[b - 1][1] - trade_amount
        trading_record.append([buy_sorted[b - 1][0], sell_sorted[s - 1][0], trade_amount, trade_price])
        if buy_sorted[b - 1][1] <= 1e-9:
            b = b + 1
        sell_sorted[s - 1][1] = sell_sorted[s - 1][1] - trade_amount
        if sell_sorted[s - 1][1] <= 1e-9:
            s = s + 1
        if b > len(buy_sorted) or s > len(sell_sorted):
            break

    return sell_sorted, buy_sorted, trading_record


def volume_preserving_carbon_aware_CDA(
    trade_re,
    P_re,
    carbon_re,
    trade_com,
    P_com,
    carbon_com,
    trade_in,
    P_in,
    carbon_in,
    carbon_price=0.05,
):
    """Prefer low-carbon sellers without sacrificing standard CDA volume."""
    standard_sell, standard_buy, standard_records = CDA(
        trade_re, P_re, trade_com, P_com, trade_in, P_in
    )
    standard_volume = sum(float(record[2]) for record in standard_records)

    participants = [
        [1, trade_re, P_re, carbon_re],
        [2, trade_com, P_com, carbon_com],
        [3, trade_in, P_in, carbon_in],
    ]
    sellers = []
    buyers = []
    for agent_id, quantity, price, carbon in participants:
        if abs(quantity) <= 1e-9:
            continue
        if quantity > 0:
            effective_price = price + carbon_price * max(carbon, 0.0)
            sellers.append([agent_id, float(quantity), price, carbon, effective_price])
        else:
            buyers.append([agent_id, float(abs(quantity)), price, carbon])

    buyers.sort(key=lambda item: item[2], reverse=True)
    records = []
    buyer_index = 0
    while buyer_index < len(buyers) and sellers:
        buyer = buyers[buyer_index]
        eligible = [
            index for index, seller in enumerate(sellers)
            if seller[1] > 1e-9 and buyer[2] + 1e-12 >= seller[2]
        ]
        if not eligible:
            # Later buyers have no higher bid, so no remaining seller is feasible.
            break
        seller_index = min(
            eligible,
            key=lambda index: (sellers[index][4], sellers[index][2], sellers[index][0]),
        )
        seller = sellers[seller_index]
        quantity = min(buyer[1], seller[1])
        price = (buyer[2] + seller[2]) / 2.0
        records.append([buyer[0], seller[0], quantity, price])
        buyer[1] -= quantity
        seller[1] -= quantity
        if seller[1] <= 1e-9:
            sellers.pop(seller_index)
        if buyer[1] <= 1e-9:
            buyer_index += 1

    volume = sum(float(record[2]) for record in records)
    if volume + 1e-9 < standard_volume:
        return standard_sell, standard_buy, standard_records
    return sellers, buyers[buyer_index:], records


def carbon_dividend_settlement(
    trading_record,
    carbon_intensities,
    grid_imports,
    grid_emission_factor,
    carbon_price,
    agent_ids=(1, 2, 3),
    dividend_mode="targeted",
):
    """Return an hourly budget-balanced carbon levy and low-carbon dividend."""
    agent_ids = tuple(agent_ids)
    if dividend_mode not in ("targeted", "equal"):
        raise ValueError(f"Unsupported carbon dividend mode: {dividend_mode}")
    price = max(float(carbon_price), 0.0)
    grid_factor = max(float(grid_emission_factor), 0.0)
    levies = {
        agent_id: price * grid_factor * max(float(grid_imports.get(agent_id, 0.0)), 0.0)
        for agent_id in agent_ids
    }
    fund = float(sum(levies.values()))

    avoided_by_seller = {agent_id: 0.0 for agent_id in agent_ids}
    for record in trading_record:
        seller = int(record[1])
        quantity = max(float(record[2]), 0.0)
        seller_intensity = max(float(carbon_intensities.get(seller, 0.0)), 0.0)
        avoided_by_seller[seller] += quantity * max(grid_factor - seller_intensity, 0.0)

    avoided_total = float(sum(avoided_by_seller.values()))
    if fund <= 1e-12:
        credits = {agent_id: 0.0 for agent_id in agent_ids}
    elif dividend_mode == "targeted" and avoided_total > 1e-12:
        credits = {
            agent_id: fund * avoided_by_seller[agent_id] / avoided_total
            for agent_id in agent_ids
        }
    else:
        equal_dividend = fund / len(agent_ids)
        credits = {agent_id: equal_dividend for agent_id in agent_ids}

    # Assign floating-point residue deterministically so the hourly fund closes exactly.
    residue = fund - float(sum(credits.values()))
    credits[agent_ids[0]] += residue
    return levies, credits, avoided_by_seller


def profit(trading_record):

    trade_profit = {1: 0, 2: 0, 3: 0}

    if len(trading_record) != 0:
        for record in trading_record:
            buyer = record[0]
            seller = record[1]
            quantity = record[2]
            price = record[3]
            trade_amount = quantity * price
            trade_profit[buyer] -= trade_amount
            trade_profit[seller] += trade_amount

    return trade_profit
