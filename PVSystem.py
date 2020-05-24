from CircuitComponents import DSS
import numpy as np
import pandas as pd


class PVSystem:
    def __init__(self, pv_list, root='Irradiance_Profile'):
        if len(pv_list) != 0:
            self.pv_list = pv_list
            self.irr_time_series = {}
            self.root = root

    def set_profile(self, day=None):
        if day is None:
            day = '20160620'

        file_name = self.root + '\Timeseries_47.405_8.505_SA_200kWp_crystSi_0_36deg_7deg_2016_2016.csv'
        self.kW = float(file_name.split("kWp")[0].split('_')[-1])
        data = pd.read_csv(file_name)
        day_time = data['time'].str.split(':', n=1, expand=True)
        data['day'] = day_time[0]
        data['hour'] = day_time[1]
        data.drop(columns=['time'], inplace=True)
        selected_profile = data.loc[data['day'] == day]

        minute_array = [60 * x + 10 for x in range(24)]
        power_array = selected_profile['P'].values / self.kW / 1000
        temperature_array = selected_profile['T2m'].values
        power_curve = np.polyfit(minute_array, power_array, 12)
        temperature_curve = np.polyfit(minute_array, temperature_array, 12)

        self.power_poly = np.poly1d(power_curve)
        self.temperature_poly = np.poly1d(temperature_curve)

        sunlight_hours = np.where(power_array > 0)[0]
        self.sunrise = sunlight_hours[0] - 0.5
        self.sunset = sunlight_hours[-1] + 0.5

    def load_pv(self, circuit, time_step):
        power = self.power_poly(time_step)
        if time_step < self.sunrise * 60 + 10 or time_step > self.sunset * 60 + 10:
            power = 0

        for x in self.pv_list:
            circuit.SetActiveElement('PVSystem2.' + x)
            circuit.ActiveCktElement.Properties('irradiance').Val = power
            circuit.ActiveCktElement.Properties('temperature').Val = self.temperature_poly(time_step)

if __name__ == '__main__':
    p = PVSystem(['PV1'])
    p.set_profile()