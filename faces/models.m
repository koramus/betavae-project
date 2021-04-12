[model msz] = load_model();

alpha = zeros(msz.n_shape_dim, 1)
beta = zeros(msz.n_tex_dim, 1)

#azimuth = 0.3 # -1 to 1
#elevation = 0 # -20 to 20
#light_azi = 180 # 90 to 270
#age = -40 # -40 to 100
#gender = 1 # -3 to 3

load ../04_attributes.mat


for age = -40:20:100
  for gender = -3:3
    shape = coef2object( alpha + age*age_shape(1:msz.n_shape_dim) + gender*gender_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
    tex   = coef2object( beta  + age*age_tex(1:msz.n_tex_dim)     + gender*gender_tex(1:msz.n_tex_dim),     model.texMU,   model.texPC,   model.texEV );

    filename = sprintf('/tmp/output/%d_%d.ply', age, gender)
    plywrite(filename, shape, tex, model.tl );
  endfor
endfor

##filename = sprintf('output.jpg')
##print(filename,'-djpeg')
