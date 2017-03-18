from __future__ import print_function

import time
from magic import gamma

if __name__ == "__main__":
   
   
    progetto_ia = gamma(
                            target_name = 'class', 
                            gamma = 'g', 
                            hadron = 'h', 
    )

    progetto_ia.prepare_data('data/magic_gamma_telescope.csv')

    start_time = time.time()
    progetto_ia()
    print("--- %s seconds ---" % (time.time() - start_time))



