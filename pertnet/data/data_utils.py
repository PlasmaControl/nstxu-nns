import numpy as np
import MDSplus as mds
import pickle
import json
import matplotlib.pyplot as plt

def save_data(save_dict, PIK = "pickle.dat"):    
		with open(PIK, 'wb') as f:
				pickle.dump(save_dict,f)

def load_data(PIK="pickle.dat"):
		with open(PIK, "rb") as f:
				load_dict = pickle.load(f)
		return load_dict

def get_signal(connection, tree, shot, tag):
		# get 1D signal from MDSplus tree
		connection.openTree(tree, shot)
		data = connection.get(tag).value
		time = connection.get('dim_of(' + tag + ')').value
		connection.closeAllTrees()
		return time, data

def get_ip(connection, tree, shot, ip_threshold=200000):
		t, ip = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:IPMEAS')
		mask = np.where(ip>ip_threshold)
		return ip[mask], t[mask]

def get_times(connection, tree, shot, ip_threshold=200000):
		t, ip = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:IPMEAS')
		mask = np.where(ip>ip_threshold)
		return t[mask]

def get_rzgrid(connection,tree,shot):
		t, rgrid = get_signal(connection, tree, shot, '.RESULTS.GEQDSK:RGRID')
		t, zgrid = get_signal(connection, tree, shot, '.RESULTS.GEQDSK:ZGRID')
		return rgrid[0], zgrid[0]

def get_conductor_data(connection,tree,shot,ip_threshold=200000):
		t,ccefit = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:CCBRSP')
		t,ecefit = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:ECCURT')
		t, ip = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:IPMEAS')
		mask = np.where(ip>ip_threshold)
		ip = ip[mask]
		coil_currents = np.zeros((len(mask[0]),13))
		vessel_currents = np.zeros((len(mask[0]), 40))
		coil_currents[:, 0] = ecefit[mask]#OH
		coil_currents[:, 1] = ccefit[mask, 0]#PF1AU
		coil_currents[:, 2] = ccefit[mask, 1]#PF1BU
		coil_currents[:, 3] = ccefit[mask, 2]#PF1CU
		coil_currents[:, 4] = ccefit[mask, 3]#PF2U
		coil_currents[:, 5] = ccefit[mask, 4]#PF3U
		coil_currents[:, 6] = (ccefit[mask, 5]+ccefit[mask, 8])/2.0#PF4
		coil_currents[:, 7] = (ccefit[mask, 6]+ccefit[mask, 7])/2.0#PF5
		coil_currents[:, 8] = ccefit[mask, 9]#PF3L
		coil_currents[:, 9] = ccefit[mask, 10]#PF2L
		coil_currents[:, 10] = ccefit[mask, 11]#PF1CL
		coil_currents[:, 11] = ccefit[mask, 12]#PF1BL
		coil_currents[:, 12] = ccefit[mask, 13]#PF1AL
		vessel_currents[:,:] = ccefit[mask,14:]   
		
		return coil_currents,vessel_currents

def get_pq_data(connection,tree,shot,ip_threshold=200000):
		t,q = get_signal(connection, tree, shot, '.RESULTS.GEQDSK:QPSI')
		t, q = get_signal(connection, tree, shot, '.RESULTS.GEQDSK:FFPRIM')
		t,p = get_signal(connection, tree, shot, '.RESULTS.GEQDSK:PPRIME')
		t, ip = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:IPMEAS')
		p = p[np.where(ip>ip_threshold)]
		q = q[np.where(ip>ip_threshold)]    
		return p, q

def get_run_data(connection,tree,shot):
		t,psirz = get_signal(connection, tree, shot, '.RESULTS.GEQDSK:PSIRZ')
		t,psia = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:PSI0')
		t,psib = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:PSIBDY')
		data = psirz.reshape(t.shape[0],psirz.shape[1]*psirz.shape[2])    
		return t,data.T, psia, psib

def get_psi_data(connection,tree,shot,ip_threshold=200000):
		t, psirz_flat, psia, psib = get_run_data(connection,tree,shot)
		t, ip = get_signal(connection, tree, shot, '.RESULTS.AEQDSK:IPMEAS')
		psirz_flat = psirz_flat.T
		psirz_flat = psirz_flat[np.where(ip > ip_threshold)].T
		return psirz_flat

def find_plasma_shots(connection, shotrange, efit_use='efit01', ip_thresh=0.5 * 10 ** 6,
											num_slices_needed_above_ip_thresh=10,
											ignored_shots=[202423, 204321, 141598, 141863, 141893, 141932, 141939, 142367]):
		plasmashots = []

		for shot in shotrange:
				try:
						time, ip = get_signal(connection, efit_use, shot, '.RESULTS.AEQDSK:IPMEAS')
						if (ip > ip_thresh).sum() > num_slices_needed_above_ip_thresh:
								if shot not in ignored_shots:
										plasmashots.append(shot)
										print('Adding to shotlist, shot ' + str(shot))
				except:
						pass
		return plasmashots
