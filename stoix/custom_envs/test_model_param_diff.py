import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_string("""
 <mujoco>
   <option gravity="0 0 -9.81"/>
   <worldbody>
     <body>
       <freejoint/>
       <geom size=".15" pos="0 0 5" mass="1" type="sphere"/>
     </body>
   </worldbody>
 </mujoco>
""")
model.opt.timestep = 0.042
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

def loss(gravity):
    mjx_model_new_gravity  = mjx_model.opt.gravity.at[2].set(gravity)
    newOpt = mjx_model.opt.replace(gravity = mjx_model_new_gravity)
    newModel = mjx_model.replace(opt = newOpt)
    mjx_data = mjx.make_data(mjx_model)
    mjx_data_new  = mjx.step(newModel, mjx_data)
    return mjx_data_new.qvel[2]

loss_grad = jax.grad(loss)

print(loss_grad(-10.0)) #-> Gives 0.042
print(loss_grad(-20.0)) #-> Gives 0.042
