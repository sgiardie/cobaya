from cobaya.likelihood import Likelihood
import numpy as np

class v_like(Likelihood):

    def initialize(self):
        """
        Importing data file cl_file set in .yaml file
        cl_file can be Vmodes_CLASS.txt or spider(150/95).txt
        """
        self.data = np.loadtxt(self.cl_file, skiprows = 1)
        
        #binning for CLASS data: bin center from file
        #each bin is 12 ell large
        if self.data.shape[1] == 3:
            #self.bin_center = self.data[:,0]
            self.bin_min = self.data[:,0].astype('int')-6*np.ones(self.data[:,0].size,dtype = int)
            self.bin_max = self.data[:,0].astype('int')+5*np.ones(self.data[:,0].size,dtype = int)
            self.ClVV_data = self.data[:,1]
            self.ClVV_err = self.data[:,2]
        
        #other class data file:
        if self.data.shape[1] == 4:
            self.bin_min = self.data[:,0].astype('int')-5*np.ones(self.data[:,0].size,dtype = int)
            self.bin_max = self.data[:,0].astype('int')+6*np.ones(self.data[:,0].size,dtype = int)
            self.ClVV_data = self.data[:,1]
            self.ClVV_err = self.data[:,2]


        #spider data files
        elif self.data.shape[1] == 6:
            self.bin_min = self.data[:,0]
            #self.bin_center = self.data[:,1]
            self.bin_max = self.data[:,2]
            self.ClVV_data = self.data[:,3]
            #self.ClVV_perr = self.data[:,4]
            #self.ClVV_nerr = self.data[:,5]
            self.ClVV_err = (self.data[:,4]-self.data[:,5])/2

        else:
            print("Error: not using the correct spider or Class data!")

       
        self.log.info("Initialized!")

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed

         e.g. here we need C_L^{vv} to lmax=2500 and the H0 value
        """
        return {'Cl': {'vv': self.bin_max[-1]}} 

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        ClVV_th = self.provider.get_Cl(ell_factor = False)['vv']
        return self.log_likelihood(ClVV_th, **params_values)

    def log_likelihood(self, ClVV_th, **params_values):
        """
        Actual calculation of the log likelihood
        """
        chi2 =  0
        for i in range(self.data.shape[0]):
            ClVV_th_bin = ClVV_th[self.bin_min[i]:self.bin_max[i]].sum()/(self.bin_max[i]-self.bin_min[i]+1)
            chi2 += (ClVV_th_bin - self.ClVV_data[i])**2/(self.ClVV_err[i])**2
        
        return -chi2/2.


