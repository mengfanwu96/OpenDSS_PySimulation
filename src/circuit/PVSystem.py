from .CircuitComponents import DSS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class PVSystem:
    def __init__(self, dss:DSS, num_steps, cloud_cover=None):
        if len(dss.pv_dict.keys()) != 0:
            self.pv_phase_dict = dss.pv_dict
            self.power_log = {}
            self.var_perturbation_denominator = 4
            if cloud_cover is None:
                self.cloud_cover = Cloud_covering(num_steps, 0.03, 3)
            else:
                self.cloud_cover = cloud_cover
            self.simulation_time = num_steps

            for x in self.pv_phase_dict.keys():
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

    def set_profile(self, root: str, day=None,
                    profile_path='/profiles/Irradiance_Profile/Timeseries_47.405_8.505_SA_200kWp_crystSi_0_36deg_7deg_2016_2016.csv'):
        if day is None:
            day = '20160620'
        abs_profile_path = root + profile_path

        self.kW = float(abs_profile_path.split("kWp")[0].split('_')[-1])
        data = pd.read_csv(abs_profile_path)
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
        self.sunrise = (sunlight_hours[0] - 0.5) * 60 + 10
        self.sunset = (sunlight_hours[-1] + 0.5) * 60 + 10

    def load_pv(self, circuit, time_step):
        power = self.power_poly(time_step)
        if time_step < self.sunrise or time_step > self.sunset or power < 0:
            power = 0

        cloud_covered_irradiance = power * (1 - self.cloud_cover.covering[time_step])
        for x in self.pv_phase_dict.keys():
            circuit.SetActiveElement('PVSystem2.' + x)
            circuit.ActiveCktElement.Properties('irradiance').Val = cloud_covered_irradiance
            circuit.ActiveCktElement.Properties('temperature').Val = self.temperature_poly(time_step)
            # TODO: panel temperature can be much higher than environment temperature

    def set_profile_short(self):
        second_array = [0, 1]
        power_array = [0.95, 0.95]
        power_curve = np.polyfit(second_array, power_array, 1)
        temperature_curve = np.polyfit(second_array, power_array, 1)

        self.power_poly = np.poly1d(power_curve)
        self.temperature_poly = np.poly1d(temperature_curve)

        self.sunrise = 0
        self.sunset = self.simulation_time - 1

    def record_pv(self, circuit, time_step):
        if time_step > 600:
            a = 1
        for x in self.pv_phase_dict.keys():
            phase_list = self.pv_phase_dict[x]
            circuit.SetActiveElement('PVSystem2.' + x)
            cplx_power = circuit.ActiveCktElement.Powers
            for i in range(len(phase_list)):
                self.power_log[x][phase_list[i]][time_step] = -complex(float(cplx_power[i]), float(cplx_power[i+1]))
                # TODO: modify the previous complex number logging, directly using function: complex()

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


class Cloud_covering:
    def __init__(self, num_steps, p=0.05, duration_avg=3):
        np.random.seed(seed=1)
        self.covering = np.random.binomial(1, p, num_steps).astype(float)
        self.num_steps = num_steps
        cloud = np.where(self.covering == 1)[0]
        for pos in cloud:
            duration = np.random.chisquare(duration_avg)
            block = int(duration) + 1
            peak_block = np.random.normal(0.75, 0.04)
            dist_peak = np.abs(np.arange(block) - float(block - 1) / 2)
            coef = np.log(2) / 2 / duration
            cover = (2 - np.exp(coef * dist_peak)) * peak_block

            cloud_end = min([num_steps, pos + block])
            self.covering[pos: cloud_end] = cover[0:cloud_end - pos]

    def set_abrupt_covering(self):
        time_array = [0, 10, 30, 40, 118, 130, 179]
        cover_array = [0.05, 0.05, 0.05, 0.8, 0.85, 0.08, 0.08]
        time_span = np.arange(self.num_steps)
        self.covering = np.interp(time_span, time_array, cover_array)

    def plot_covering(self):
        plt.plot(self.covering)
        plt.show()



if __name__ == '__main__':
    p = PVSystem(['PV1'])
    p.set_profile()