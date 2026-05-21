import pandas as pd
import random

data = []

for i in range(1000):

    distance = random.randint(5000, 30000)

    temperature = random.randint(60, 130)

    oil_level = random.randint(10, 100)

    vibration = random.randint(1, 12)

    # Maintenance Logic
    if (

        temperature > 105

        or oil_level < 40

        or vibration > 7

        or distance > 20000

    ):

        maintenance = 1

    else:

        maintenance = 0

    data.append([

        distance,
        temperature,
        oil_level,
        vibration,
        maintenance

    ])

df = pd.DataFrame(

    data,

    columns=[

        'Distance',
        'Temperature',
        'Oil_Level',
        'Vibration',
        'Maintenance'

    ]

)

df.to_csv("transport_data.csv", index=False)

print("✅ 1000 Rows Dataset Generated Successfully")