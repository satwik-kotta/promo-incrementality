# Do Grocery Deals Pay for Themselves?

Retailers know a deal sells more units. What they rarely know is how many of those units were new demand, and how many were sold at a discount to shoppers who would have bought anyway. This project answers that for 5,143 real grocery deals, then costs each one.

**Live dashboard:** https://satwik-kotta.github.io/promo-incrementality/

**One page memo for a category manager:** [MEMO.md](MEMO.md)

Data: the 84.51° (Kroger) Complete Journey. 1,469,307 transactions from 2,469 loyalty card households over 2017, with weekly mailer and in store display records for every product and store.

## What it found

| Question | Answer |
|---|---|
| How much of a deal is new demand? | **41%** of units sold on deal would have sold at the regular price that week anyway. Of the extra units, 23% were taken from other products in the same category. |
| How many deals make money for the retailer? | **4.0%** at a 30% gross margin, before any manufacturer funding. The median deal breaks even only at a 68% margin. |
| What would make the median deal break even? | The manufacturer funding **62%** of the discount, at a 30% margin. |
| Does the mailer matter? | Yes, more than anything else. Being in the weekly mailer adds **+46%** units (95% CI 40 to 54%) on top of the price cut. Shelf only price cuts need a 96% margin to break even and 61% of their units would have sold anyway. Mailer deals need 63%. |
| How price sensitive is demand? | Elasticity of **−1.89**. A 25% discount sells about 1.7 times the units. |
| Do deals just pull sales forward? | No detectable dip. Holding price fixed, weeks 2 to 4 after a deal sell within a few percent of a normal week (−2.4% to +2.6%, every interval includes zero). |
| Does the lift fade over a long deal? | No. Weeks 2 and 3 of a deal sold about 4% more than week 1, not less. |
| Is it a loss leader? | A little. Each deal on an item a household buys regularly adds **$0.23** of spending on other items that week (95% CI $0.03 to $0.43) and 0.02 extra trips. A placebo using next week's deals shows nothing. Counting this halo, **28%** of deals make money at a 30% margin (range 5% to 50%), and deals as a whole still lose money. |
| Do deals create lasting customers? | Deals bring in new buyers (+6.3 points in the share of purchases from households new to the product), but new buyers who found the product on deal were **22% less likely** to buy it again at full price within 8 weeks (14.0% vs 18.0%). |

The headline is the gap between the units a deal sells and the units it adds. Most deals look successful on a sales report and lose money once the discount on baseline buyers and the cannibalized sales are counted.

## Recommendations

1. **Stop running price cuts that are not advertised.** They mostly discount shoppers who were going to buy anyway.
2. **Get vendor funding agreed before committing to a deal.** Without it, almost no deal in this data pays back.
3. **Do not cap deals on the assumption that the lift fades.** It did not fade here.
4. **Review deals on category and basket profit.** About a quarter of the lift came from sister products, and the basket halo recovers part of the loss, but not enough to make deals pay overall.
5. **Do not count deal buyers as acquired customers.** They are less likely than other new buyers to return at full price.

## Method

**Panel.** One row per product and week for the 1,260 products with at least 200 units, 45 weeks of sales and 30 buyers (fuel and coupon lines excluded), weeks 2 to 52 (weeks 1 and 53 are partial). Zero sales weeks are kept as zeros. `sql/01_panel.sql`.

**What counts as a deal.** The data has two kinds of promotion: the weekly mailer and in store displays, and the loyalty card price discount. Mailers and displays mark a deal week, but price cuts also happen in weeks with no mailer. So a deal week is defined by price: the loyalty discount is at least 15% of shelf price. 34% of product weeks are deal weeks. Results hold at 10% and 20% (see Robustness).

**Lift model.** Poisson regression of weekly units with product fixed effects and department by week fixed effects, standard errors clustered by product (`src/model.py`, pyfixest):

```
units ~ log net price + mailer + display + share of same category products on deal
        + week 2 of deal + week 3+ of deal
        + 1, 2, 3, 4 weeks after a deal + 1, 2 weeks before a deal
        | product + department x week
```

Department by week fixed effects absorb seasonality and holidays. Product fixed effects mean each product is compared with itself. Controlling for price lets the before and after terms show whether demand dips once the price goes back up.

**Counterfactual.** For each deal week, the model gives the units the product would have sold at its regular discount with no mailer. That is the baseline. Units above it are the lift.

**Cannibalization.** The coefficient on the share of same category products on deal (−16% when every sister product is on deal) gives the units each deal took from its category that week.

**Profit.** For a gross margin m on the regular price, the change in profit is the change in net revenue, minus the cost of the extra units, minus the margin lost on cannibalized units. This is linear in m, so each deal has an exact break even margin. Manufacturer funding is reported as the share of the discount it would take to break even at 30%.

**Basket halo.** Each household's regular products are the in scope products it bought in 3 or more weeks. The number of them on deal varies from week to week and differs across households, because regular baskets differ. Linear models with household and week fixed effects estimate how one more deal on a household's regular items changes its trips and its spending on items not on deal. Next week's deal count is included as a placebo. The halo is then added to each deal at the assumed margin, using the number of households that buy that product regularly (`src/halo.py`). Because spending on other items already nets out sister products losing sales, this adjustment is conservative.

**New buyers.** Each household purchase from week 15 to 44 is flagged new if the household had not bought the product in the previous 13 weeks, then followed for 8 weeks. Linear probability models with product and week fixed effects compare deal and regular weeks. The return model also controls for how many full price weeks followed.

## Robustness

| Deal threshold | Elasticity | Mailer effect | Would have sold anyway | Profitable at 30% | Median break even |
|---|---|---|---|---|---|
| 10% | −1.90 | +47% | 43% | 5.0% | 66% |
| 15% (main) | −1.89 | +46% | 41% | 4.0% | 68% |
| 20% | −1.88 | +45% | 40% | 2.7% | 72% |

Tests (`tests/`) simulate a panel with known price and mailer effects and no dip, and check that the model recovers them. They also check the episode logic and that profit is exactly zero at the break even margin.

## Limits

- **Panel only.** These are 2,469 loyalty card households, not every shopper, so dollar totals are for the panel.
- **Margin and funding are assumed.** Neither is in the data. The dashboard lets you pick a margin, and funding is reported as the share needed to break even.
- **Week boundaries.** Deals start and end partway through a week, so the week before and after a deal carry some of it. That is why both show small positive effects (+5.9% and +5.2%).
- **Halo window.** The halo is measured in the same week. Effects on loyalty over months are not counted. Cannibalization is only within the same category and week.
- **Association for new buyers.** The new buyer comparison is within product and week, but households that wait for deals may differ from those that do not.

## Run it

```bash
pip install -r requirements.txt
make all        # download data, build the panel, fit the model, write outputs and docs/data.js
make test
```

Takes about 15 seconds after the download.

```
sql/01_panel.sql         product x week panel
src/get_data.py          download the Complete Journey tables
src/features.py          deal episodes, weeks before and after, category deal share
src/model.py             lift model, counterfactual, deal profit and break even margin
src/halo.py              basket halo and halo adjusted deal profit
src/run_all.py           pipeline, household checks, halo, robustness, outputs/results.json
src/export_dashboard.py  docs/data.js for the dashboard
docs/                    dashboard (GitHub Pages)
outputs/                 lift_effects.csv, deal_episodes.csv, results.json
```

Data source: 84.51° Complete Journey, via the open source [completejourney](https://github.com/bradleyboehmke/completejourney) package by Brad Boehmke.
