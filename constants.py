input_shape = [132, 132, 132]
output_shape = [44, 44, 44]

extra = 0
input_shape = list(x + extra * 8 for x in input_shape)
output_shape = list(x + extra * 8 for x in output_shape)

fmap_start = 24
unet_fmap_decrease_factor = 3
unet_fmap_increase_factor = 3
