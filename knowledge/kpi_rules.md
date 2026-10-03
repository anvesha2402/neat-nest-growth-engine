# Neat Nest KPI Definitions and Review Policies

## [KPI_NET_SALES] Net sales
Net sales is revenue after discounts and excluding GST, summed over order lines. All values are in INR. Returns are not deducted in the sales dataset. Net sales is the default metric when a question does not name one.

## [KPI_GROSS_PROFIT] Gross profit
Gross profit equals net sales minus product cost, where product cost is unit COGS plus packaging. It excludes shipping, channel fees, returns and marketing, which are covered by contribution margin.

## [KPI_GROSS_MARGIN] Gross margin
Gross margin equals gross profit divided by net sales, multiplied by 100. A change in margin is reported in percentage points. When net sales are zero the margin is undefined.

## [KPI_CONTRIBUTION] Contribution margin (CM1, CM2, CM3)
CM1 is net revenue minus COGS and packaging. CM2 subtracts shipping, payment or marketplace fees and a returns allowance. CM3 subtracts customer acquisition cost. Amazon Home Storage charges a 13% referral fee on items above Rs 300 plus a Rs 56 closing fee above Rs 1,000.

## [KPI_ATTACH] Accessory attach rate and lapse
Attach rate is accessory units divided by consumer core units in the same period. A customer is lapsed when, in months 7-12 after purchase, they make no accessory purchase, no referral and no care-hub login. Lapse replaces "churn" because FlexFold is a durable product that is not repurchased each month.

## [KPI_CAC] Customer acquisition cost and conversion
CAC is paid acquisition spend divided by new customers acquired through that channel. Campaign conversion is purchasers divided by prospects reached. Amazon advertising efficiency is tracked as ACoS, ad spend divided by attributed sales.

## [POLICY_DECLINE] Sales decline review
When a city or channel declines month on month, first split the change by product and by channel, then check units against price per unit. If units fall while price per unit holds, check stock availability and listing status first. If price per unit falls, review discounting. Check competitor activity and local events last. A data split shows where a decline occurred, not why it occurred.

## [POLICY_STOCKOUT] Stock-out and heritage sleeve review
Heritage Edition sleeves are printed in regional batches, so a single sleeve can run out in one city while the core product is in stock. If Heritage Edition units in a city fall by more than half month on month, check sleeve inventory for that city's lead craft before any other cause. A Heritage stock-out often softens core sales too, because the sleeve is the main draw on the city landing page.

## [POLICY_DISCOUNT] Discount guardrails
FlexFold may not be discounted by more than 10% on D2C or 15% on marketplaces. Neat Nest does not join 40-80% marketplace sale events, because deep discounting undermines the durability positioning. If the average discount in a city or channel rises by more than 3 percentage points month on month, review it against gross margin.

## [DATA_GUIDE] Data guide for the copilot
The sales dataset is a synthetic simulation of Neat Nest's launch year, April 2027 to March 2028. Mumbai and Pune trade from April 2027; Bengaluru, Hyderabad and Delhi NCR from October 2027. Channels are D2C, Amazon and B2B. Products are FlexFold Lite, FlexFold, FlexFold Heritage Edition, FlexFold Pro (B2B), Liner Set, Charcoal Odour Insert and Wheel Kit. It does not contain stock, traffic, competitor or customer-level data.
