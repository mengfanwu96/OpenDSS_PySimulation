from CircuitComponents import DSS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PVSystem:
    def __init__(self, dss:DSS, num_steps, root='Irradiance_Profile'):
        if len(dss.pv_dict.keys()) != 0:
            self.pv_phase_dict = dss.pv_dict
            self.root = root
            self.power_log = {}
            self.var_perturbation_denominator = 4

            for x in self.pv_phase_dict.keys():
                phase_list = []
                dss.circuit.SetActiveElement('PVSystem2.' + x)
                phase_num = int(dss.circuit.ActiveCktElement.NumPhases)

                if phase_num == 3:
                    phase_list = [1, 2, 3]
                else:
                    phases = dss.circuit.ActiveCktElement.Properties('bus1').Val.split('.')
                    phases.pop(0)
                    phase_list = [int(x) for x in phases]
                    phase_list.sort()

                self.pv_phase_dict[x] = phase_list
                self.power_log[x] = {}
                for i in phase_list:
                    self.power_log[x][i] = np.zeros(num_steps, dtype=complex)


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
        if time_step < self.sunrise * 60 + 10 or time_step > self.sunset * 60 + 10 or power < 0:
            power = 0

        perturbed = np.random.normal(loc=power, scale=power/self.var_perturbation_denominator)
        perturbed = max([perturbed, 0])
        for x in self.pv_phase_dict.keys():
            circuit.SetActiveElement('PVSystem2.' + x)
            circuit.ActiveCktElement.Properties('irradiance').Val = power
            circuit.ActiveCktElement.Properties('temperature').Val = self.temperature_poly(time_step)
            # TODO: panel temperature can be much higher than environment temperature

    def record_pv(self, circuit, time_step):
        if time_step > 600:
            a = 1
        for x in self.pv_phase_dict.keys():
            phase_list = self.pv_phase_dict[x]
            circuit.SetActiveElement('PVSystem2.' + x)
            cplx_power = circuit.ActiveCktElement.Powers
            for i in range(len(phase_list)):
                self.power_log[x][phase_list[i]][time_step] = -complex(float(cplx_power[i]), float(cplx_power[i+1]))
                # TODO: modify the previous complex number logging

    def plot_pv(self, pv_list=None):
        # TODO: selective plotting
        for x, phase_list in self.pv_phase_dict.items():
            fig, axes = plt.subplots(len(phase_list), 1)
            if len(phase_list) == 1:
                axes = [axes]

            for idx, ax in enumerate(axes):
                ax.plot(np.abs(self.power_log[x][phase_list[idx]]))
                ax.set_xlabel('time (m)')
                ax.set_ylabel('power')

            fig.show()

if __name__ == '__main__':
    p = PVSystem(['PV1'])
    p.set_profile()