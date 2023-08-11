<template>
  <div class="home">
    <div class="common-layout">

      <el-container>
      <el-container>
        <el-header>
          <h1 id="MISCSIM23">Misc Physics Simulation Demoes</h1>
        </el-header>
        <el-container>
          <el-container>
            <el-main>
              <el-row class="row-bg"><br/></el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  This page includes some of miscellaneous physics simulation demoes written in 2022 ~ 2023. Some of them
                  are personal projects, some of them are course projects of <el-link href="https://cseweb.ucsd.edu/~alchern/teaching/cse291_sp23//" target="_blank" type="primary">CSE 291</el-link>
                  instructed by Professor <el-link href="https://cseweb.ucsd.edu/~alchern/" target="_blank" type="primary">Albert Chern</el-link>.
                </p>
              </el-row>

              <!-- Chapter 1: Introduction -->
              <el-row class="row-bg">
                <h2 id="1. Introduction" style="text-align:left">1. PBD Rigid Body by Shape Matching</h2>
                <p style="text-align:left">
                  Implement PBD (Position Based Dynamics) rigid body collision. Every particle moves and collides individually,
                  and then use shape matching to recover the shape of the rigid body.
                </p>
              </el-row>
              
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The implementation is writen in C++ (host) and visualized by OpenGL:
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                  <el-col :span="17">
                    <iframe width="560" height="315" src="https://www.youtube.com/embed/ehA1nLbxIGw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
                  </el-col>
                </el-row>

              <!-- Chapter 2: BDPT & MMLT -->
              <el-row class="row-bg" justify="left">
                <h2 id="2.">2. Double Pendulum</h2>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Deriving double pendulum kinetic energy and potential energy. Using optimality for least action, we could write down the ODE of double pendulum.
                  Then I use RK4 to solve the ODE and get the trajectory of the double pendulum.
                </p>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  To keep the result simple let's just say:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <math-jax latex="\begin{aligned}
                  (\frac{5}{3}+\cos\phi) \ddot{\theta} + (\frac{1}{3}+\frac{1}{2}\cos\phi)\ddot{\phi}=A\\
                  (\frac{1}{3}+\frac{1}{2}\cos\phi)\ddot{\theta} + \frac{1}{3}\ddot{\phi}= B
                  \end{aligned}
                  " />.
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  To further simplify, we observe that:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <math-jax latex="\begin{aligned}
                  C = A-B = \frac{1}{2}\sin\phi(\dot{\theta}+\dot{\phi})^2 -\frac{3g}{2l}\sin\theta\\
                  B = -\frac{g}{2l}\sin(\theta+\phi) - \frac{1}{2}\sin\phi \dot{\theta}\dot{\theta}\\
                  f = \ddot{\theta} = \frac{18\cos\phi B -  12 C}{9\cos\phi^2 - 16}\\
                  g = \ddot{\phi} = \frac{(12+18\cos\phi)C - (48+18\cos\phi)B}{9(\cos\phi)^2 -16}
                  \end{aligned}
                  " />.
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  And the final ODE is just:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <math-jax latex="\frac{d}{dt}\begin{bmatrix}
                \theta\\
                \phi\\
                \dot\theta\\
                \dot \phi
                \end{bmatrix}
                = \begin{bmatrix}
                \dot{\theta}\\
                \dot{\phi}\\
                f(\theta,\phi,\dot{\theta},\dot{\phi})\\
                g(\theta,\phi,\dot{\theta},\dot{\phi})
                \end{bmatrix}" />.
              </el-row>
              
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The implementation is based on Houdini:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                  <el-col :span="17">
                    <iframe width="560" height="315" src="https://www.youtube.com/embed/R0VZUmkZooQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
                  </el-col>
                </el-row>
              
              <!-- Chapter 2: BDPT & MMLT -->
              <el-row class="row-bg" justify="left">
                <h2 id="3.">3. Softbody 3D Tetrahedral</h2>
              </el-row>
              
                <el-row class="row-bg" justify="left">
                <h3 id="3.1.1" style="margin-top: 0in;">3.1 Per Cell Force computation</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  As all the selected mesh used in the demo are composed by tetrahedral. Our basic discrete element is tetrahedral. Let's call the rest position 
                  <math-jax latex="X_i" />, which is given by the input file data, and the curretn world position <math-jax latex="x_i"/> is variable in the simulation.
                </p>
                <p style="text-align:left">
                  In the first pass, we iterate throught all the tetrahedral, compute the force it distributes to each of its vertices.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="3.1.2" style="margin-top: 0in;">3.1.1 Deformation Gradient</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  For simplicity, we should simply consider Lagragian coordinate M as reference space and Eulerian coordinate W as actual world space here.
                </p>
                <p style="text-align:left">
                  Then the state of a deformable body is a flow map $\Phi: M\rightarrow W$. We could expect that this flow map contains the information about how much deformation is happening.
                </p>
                <p style="text-align:left">
                  If we consider coordinate. Then we could know:

                  <math-jax latex="$$
                    \begin{bmatrix}
                        x\\y\\z
                    \end{bmatrix} = \begin{bmatrix}
                        \phi^1(X,Y,Z)\\
                        \phi^2(X,Y,Z)\\
                        \phi^3(X,Y,Z)
                    \end{bmatrix}
                    $$
                    "/>
                </p>
                <p style="text-align:left">
                  Then we could get the deformation gradient <math-jax latex="F=d\phi"/> as:

                  <math-jax latex="     
                    $$
                    F=\begin{bmatrix}
                        \frac{\partial \phi^1}{\partial X} & 
                        \frac{\partial \phi^1}{\partial Y} &
                        \frac{\partial \phi^1}{\partial Z} \\
                        \frac{\partial \phi^2}{\partial X} & 
                        \frac{\partial \phi^2}{\partial Y} &
                        \frac{\partial \phi^2}{\partial Z} \\
                        \frac{\partial \phi^3}{\partial X} & 
                        \frac{\partial \phi^3}{\partial Y} &
                        \frac{\partial \phi^3}{\partial Z} \\
                    \end{bmatrix}
                    $$"/>
                </p>
                <p style="text-align:left">
                    If we consider Finite Element method, we could express the mapping as:

                    <math-jax latex="     
                    $$
                    x= FX + c
                    $$"/>
                  </p>
                <p style="text-align:left">
                    The mapping <math-jax latex="$$\phi$$"/> is a linear transform, which could be decomposed into volume change (the jacobian of the transform, <math-jax latex="$$d\phi=\frac{\partial x}{\partial X}=F$$"/>) and translation c.
                  </p>
                <p style="text-align:left">
                  As <math-jax latex="$$x_{10}=x_1-x_0=FX_1+c-FX_0-c=FX_{10}$$"/> , where we eliminate the translation term c by subtraction. Therefore we could actually derive that
                  <math-jax latex="$$F = \begin{bmatrix}
    | & | & | & 0 \\ 
    x_{10} & x_{20} & x_{30} & 0 \\ 
    | & | & | & 0 \\
    0 & 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
    | & | & | & 0 \\ 
    X_{10} & X_{20} & X_{30} & 0 \\ 
    | & | & | & 0 \\
    0 & 0 & 0 & 1
\end{bmatrix}^{-1}$$"/>
</p>
                <p style="text-align:left">
                  Another different view is using homogeneous coordinates, which extend coordinates to an extra dimension to account for transformation. Then the transformation could be expressed as :
                  <math-jax latex="$$x=FX$$"/>:
                  <math-jax latex="
$$
\begin{bmatrix}
    |\\y\\|\\1
\end{bmatrix} = \begin{bmatrix}
    | & | & | & |\\
    x_0 & x_1 & x_2 & x_3\\
    | & | & | & |\\
    0 & 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
    | & | & | & |\\
    X_0 & X_1 & X_2 & X_3\\
    | & | & | & |\\
    0 & 0 & 0 & 1
\end{bmatrix}^{-1}
\begin{bmatrix}
    |\\Y\\|\\1
\end{bmatrix}
$$"/>
</p>
                <p style="text-align:left">
                  In this setting, translation is also introduce in F, but we want to represent deformation so translation is not welcomed. We may be able to eliminate the translation by removing the edge elements:
                  <math-jax latex="
$$
\begin{bmatrix}
     &  & & 0\\
     & F & & 0\\
     & & & 0\\
     0 & 0& 0& 1
\end{bmatrix}
\leftarrow \begin{bmatrix}
    | & | & | & |\\
    x_0 & x_1 & x_2 & x_3\\
    | & | & | & |\\
    0 & 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
    | & | & | & |\\
    X_0 & X_1 & X_2 & X_3\\
    | & | & | & |\\
    0 & 0 & 0 & 1
\end{bmatrix}^{-1}
$$"/>:
</p>
                <p style="text-align:left">
                  But we should notice that the F also contains rotation, which is again not related to deformation, so we should further doing more things on F to eliminate rotation, extract the deformation representation. That is strain.

                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="3.3" style="margin-top: 0in;">3.1.2 Strain</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  In Green Strain <math-jax latex="C=F^TF"/>, we eliminate the rotation term by multiplying the transpose with itself, it could be explained by SVD decomposition. <math-jax latex="C"/> will be close to identity matrix when no deformation happened.
                </p>
                <p style="text-align:left">
                  Therefore we could further subtract C by identity and get Green-st venant <math-jax latex="E =\frac{1}{2}(C-I)"/>, which is a representation for deformation. Where when no deformation happens it will be close to zero matrix.
                </p>
                <p style="text-align:left">
                  We further use Saint Venant-Kirchhoff Model to mapping the strain to the energy density per reference area W. 
                  Then we could deriviate the energy density upon deformation and get<math-jax latex="S=\frac{\partial W}{\partial S}"/> which is 2nd Piola-Kirchhoff stress tensor.
                  <math-jax latex="S = 2\mu \textbf{E} +\lambda \textbf{tr(E)I}"/>.
                </p>
                <p style="text-align:left">
                  Have get the deformation gradient <math-jax latex="\mathbf{F}"/>, we could then do  some more steps for each cell:
                </p>
                <ol>
                  <li style="text-align:left"> Build right Cauchy-Green tensor:  <math-jax latex="\mathbf{C} = \mathbf{F}^T\mathbf{F}"/> </li>
                  <li style="text-align:left"> Build Green-st venant:  <math-jax latex="E = \frac{1}{2}(\textbf{C-I})"/> </li>
                  <li style="text-align:left"> Build 2nd Piola-Kirchhoff:  <math-jax latex="S = 2\mu \textbf{E} +\lambda \textbf{tr(E)I}"/> </li>
                  <li style="text-align:left"> Build 1st Piola-Kirchhoff:  <math-jax latex="\mathbf{P=FS}"/> </li>
                </ol>
              </el-row>

<el-row class="row-bg" justify="left">
                <h3 id="3.4" style="margin-top: 0in;">3.1.3 Forcce</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Energy density per reference area <math-jax latex="$$W(G)$$" />.
                </p>
                <p style="text-align:left">
                  Then we could know the total energy: <math-jax latex="$$E=\int W(G)dA$$" />.
                </p>
                <p style="text-align:left">
                  Given the energy we could compute the force by take derivative to the position: <math-jax latex="$$
f_i = -(\frac{\partial E}{\partial x_i})^T = -\frac{A}{n}Pn
$$" />.
                </p>
                <p style="text-align:left">
                  The answer is related to area/volume as well as normal. Interestingly it is closely related to the matrix we choose previously for computation of F. 
                  The relation could be simply reveal by noticing the compound product of <math-jax latex="$$X_{01}, X_{02}, X_{03}$$" /> and their permutations should always be the volume.
                </p>
              </el-row>



              <el-row class="row-bg" justify="left">
                <h3 id="3.4" style="margin-top: 0in;">3.2 Intermediate Connection</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  In this way we could compute the force each cell gives to each vertex. By iterating through all the cells, we could accumulate all the traction force given to a vertex by different cells.
                  <math-jax latex="$$
f_v=\frac{1}{n}\sum_{c\succ v}{\rm P}_c{\rm n}_{c,v}A_{c,v}
$$" /> 
                </p>
                <p style="text-align:left">
                  (Adjoint) All the vertices in the cell gives their force to the cell.
                  <math-jax latex="$$
                  \sum_c \textbf{tr}P_c(\mathring{F}_c^T)V_c = \sum_V-f_V^T\mathring{x}_V
$$" /> 
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="3.4" style="margin-top: 0in;">3.3 Per Vertex Equation of Motion</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Given the force, we could use the equation of motion to do time integration.
                </p>
              </el-row>
              <math-jax latex="$$
                \mathrm{m_v\ddot{x}_v=f_v}
                $$" />

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The implementation is based on
                  <el-link href="https://github.com/SuikaSibyl/SIByLEngine2023/" target="_blank" type="primary">my own Engine</el-link>,
                  the simulation code is on CPP and the visualization is on Vulkan:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                  <el-col :span="17">
                    <iframe width="560" height="315" src="https://www.youtube.com/embed/N3MmjSIYpBI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
                  </el-col>
                </el-row>
              
              <!-- Chapter 2: BDPT & MMLT -->
              <el-row class="row-bg" justify="left">
                <h2 id="2.">4. FLIP 2D Fluid</h2>
              </el-row>              
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The implementation is based on
                  <el-link href="https://github.com/SuikaSibyl/SIByLEngine2023/" target="_blank" type="primary">my own Engine</el-link>,
                  the simulation code is on CPP and the visualization is on Vulkan:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                  <el-col :span="17">
                    <iframe width="560" height="315" src="https://www.youtube.com/embed/I1V52ImMg4E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
                  </el-col>
                </el-row>
            </el-main>
            <el-divider />
            <el-footer>
              <el-link href="https://suikasibyl.github.io/" target="_blank" type="primary">My Homepage</el-link>
            </el-footer>
          </el-container>
        </el-container>
      </el-container>
    </el-container>
    </div>
  </div>
</template>

<style scoped>
.el-link {
  margin-right: 8px;
}
.el-link .el-icon--right.el-icon {
  vertical-align: text-bottom;
}
</style>

<script>

import '../assets/githubmd.css';

export default {
  name: 'ReportPage',
  mounted() {
  },
  props: {
  },
  data() {
    return {
      formula: '$$x = {-b \\pm \\sqrt{b^2-4ac} \\over 2a}.$$'
    }
  },
}
</script>
