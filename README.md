# ens_marketing_model
A model for assessing the effectiveness of Ethereum Name Service marketing, using data on external (general market) factors.

# Overview

This model uses the logic described in the [ENS marketing study](https://medium.com/@d.potyomka/ens-marketing-research-7ca38cd1c4d9). It evaluates marketing efforts in order to establish an adequate view of ongoing campaigns. Performance ratings are based on the influence of external factors, since internal ENS metrics are not available due to their confidentiality.

# Usage
The principles of operation of the model are described in the comments in the code itself.

1. Data parsing is not automated, so update file data
'[totalmarketcap.csv](https://github.com/danchousz/ens_marketing_model/blob/main/model/totalmarketcap.csv)' (ticker 'Total' in TradingView), closing prices for each week;
'[transactions.csv](https://github.com/danchousz/ens_marketing_model/blob/main/model/transactions.csv)' (from Dune query), average daily number of transactions per week;
and '[registrations.csv](https://github.com/danchousz/ens_marketing_model/blob/main/model/registrations.csv)' (from Dune query), the number of ENS name registrations.

2. Run the '[model.py](https://github.com/danchousz/ens_marketing_model/blob/main/model/model.py)' program and get the output in csv and xslx formats.

3. Analyze a specific campaign by assessing the effectiveness of marketing over the period in which the campaign was conducted.

e.g: ENS marketing effectiveness was 62% 09/27/2022 – 10/03/2022. This means that 62% of the number of registrations was justified by marketing efforts, and not by external factors, which means that the campaigns carried out at that moment were successful and this must be taken into account when forming strategies.

The opposite situation occurred from 05/01/2023 to 05/08/2023, when the number of registrations increased, but this was caused only by market movements, since the contribution of marketing was only 5%.


# Requirements
-python (3.12.0)

-pip

-[libraries](https://github.com/danchousz/ens_marketing_model/blob/main/requirements.txt)
