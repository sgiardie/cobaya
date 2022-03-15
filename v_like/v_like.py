from cobaya.likelihood import Likelihood
import numpy as np

class VLike(Likelihood):

    def initialize(self):
        """
        Importing data file cl_file set in .yaml file
        cl_file can be Vmodes_CLASS.txt or spider(150/95).txt
        """
        self.data_C = np.loadtxt(self.cl_class, skiprows = 1)
        self.data_sp150 = np.loadtxt(self.cl_sp150, skiprows = 1)
        self.data_sp95 = np.loadtxt(self.cl_sp95, skiprows = 1)

        #binning for CLASS data: bin center from file
        #each bin is 12 ell large
        self.lmin_C =  np.arange(10)*12+1
        self.lmax_C = (np.arange(10)+1)*12
        self.ClVV_data_C = self.data_C[:,1]
        self.ClVV_err_C = self.data_C[:,2]


        #spider data files
        self.lmin_sp150 = self.data_sp150[:,0].astype('int')
        self.lmax_sp150 = self.data_sp150[:,2].astype('int')
        self.ClVV_data_sp150 = self.data_sp150[:,3]
        self.ClVV_err_sp150 = (self.data_sp150[:,4]-self.data_sp150[:,5])/2

        self.lmin_sp95 = self.data_sp95[:,0].astype('int')
        self.lmax_sp95 = self.data_sp95[:,2].astype('int')
        self.ClVV_data_sp95 = self.data_sp95[:,3]
        self.ClVV_err_sp95 = (self.data_sp95[:,4]-self.data_sp95[:,5])/2

        self.log.info("Initialized!")

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed

         e.g. here we need C_L^{vv} to lmax=1000 (default in the .yaml, can be changed) 
         """
        if self.lmax >= self.lmax_sp95[-1]:
            return {'Cl': {'vv': self.lmax}}
        else:
            print("Error: lmax of theory is smaller than the lmax of data!")

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        ClVV_th = self.provider.get_Cl(ell_factor = True)['vv']
        return self.log_likelihood(ClVV_th, **params_values)

    def log_likelihood(self, ClVV_th, **params_values):
        """
        Actual calculation of the log likelihood
        """
        chi2 =  0
        for i in range(self.data_C.shape[0]):
            ClVV_th_bin_C = ClVV_th[self.lmin_C[i]:self.lmax_C[i]].sum()/(self.lmax_C[i]-self.lmin_C[i]+1)
            chi2 += (ClVV_th_bin_C - self.ClVV_data_C[i])**2/(self.ClVV_err_C[i])**2
       
        for i in range(self.data_sp150.shape[0]):
            ClVV_th_bin_sp150 = ClVV_th[self.lmin_sp150[i]:self.lmax_sp150[i]].sum()/(self.lmax_sp150[i]-self.lmin_sp150[i]+1)
            chi2 += (ClVV_th_bin_sp150 - self.ClVV_data_sp150[i])**2/(self.ClVV_err_sp150[i])**2
          
        for i in range(self.data_sp95.shape[0]):
            ClVV_th_bin_sp95 = ClVV_th[self.lmin_sp95[i]:self.lmax_sp95[i]].sum()/(self.lmax_sp95[i]-self.lmin_sp95[i]+1)
            chi2 += (ClVV_th_bin_sp95 - self.ClVV_data_sp95[i])**2/(self.ClVV_err_sp95[i])**2

        return -chi2/2.


