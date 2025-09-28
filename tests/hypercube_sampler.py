#####################################################################################
from bernet.core.sampler import HypercubeSampler

#####################################################################################
def _unidimensional_test() -> None:

    #-- Parameters
    spacing = 0.01
    dim = 1
    batch_size = 5
    ratio = 0.3
    
    #-- Sampler
    sampler = HypercubeSampler(
        spacing=spacing,
        dim=dim,
        batch_size=batch_size,
        ratio=ratio,
        solution=lambda x: x,
    )

    #-- Generate points
    num_batches = sampler.generate()
    print(f"Number of batches: {num_batches}")

    return

if __name__ == "__main__":
    _unidimensional_test()

#####################################################################################