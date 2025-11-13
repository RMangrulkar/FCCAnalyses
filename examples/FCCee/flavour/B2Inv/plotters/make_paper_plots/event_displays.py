import os
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d, art3d
from matplotlib.patches import FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
from matplotlib.legend import Legend
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
plt.rcParams['text.usetex'] = True

path = "../ella_files/for_evt_disp"

samples = { "Bd" : "Bd.root",
            "Zbb": "bb.root",
            "Zud": "ud.root", 
          }

event_choice = { "Bd": 53,
                 "Zbb": 19,
                 "Zud": 43,
               }

thrust_vars = [ "EVT_Thrust_x", "EVT_Thrust_y", "EVT_Thrust_z",
                "EVT_Thrust_xerr", "EVT_Thrust_yerr", "EVT_Thrust_zerr" ]

rec_vars = [ "Rec_n", "Rec_e", "Rec_m", "Rec_q", "Rec_p", "Rec_pt", "Rec_eta", "Rec_phi", "Rec_px", "Rec_py", "Rec_pz", "Rec_true_PDG", "Rec_indvtx" ]

vtx_vars = [ "Rec_vtx_n", "Rec_vtx_ntracks", "Rec_vtx_chi2", "Rec_vtx_isPV", "Rec_vtx_m", "Rec_vtx_x", "Rec_vtx_y", "Rec_vtx_z", "Rec_vtx_xerr", "Rec_vtx_yerr", "Rec_vtx_zerr" ] #, "Rec_vtx_indRP" ]

# class to draw an arrow in 3D plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.max(zs)

def conv_uproot_to_unit_vec( array_dict, prefix, norm=True ):

    x = array_dict[prefix+"x"][0]
    y = array_dict[prefix+"y"][0]
    z = array_dict[prefix+"z"][0]
    
    if norm:
        mag = np.sqrt( x**2 + y**2 + z**2 )
        x /= mag
        y /= mag
        z /= mag

    return np.array( [x, y, z] )

# Compute a small nudge toward the camera
def move_point_toward_camera(ax, x, y, z, delta=1e-4):
    proj = ax.get_proj()
    xyz_proj = proj3d.proj_transform(x, y, z, proj)
    # Move it slightly closer along the Z-screen axis
    return proj3d.inv_transform(xyz_proj[0], xyz_proj[1], xyz_proj[2] - delta, proj)

def draw_event_display( file, event_number, elev=None, azim=None, roll=None, clip=None, hemis_stretch=1, save=None ):

    if not os.path.exists( file ):
        raise RuntimeError( f"No file found at {file}" )
    
    stretch = 8

    xrange = (-stretch, stretch)
    yrange = (-stretch, stretch)
    zrange = (-stretch, stretch)

    # create the figure
    fig = plt.figure( facecolor='white' )
    ax = fig.add_subplot( 111, projection='3d' )
    ax.grid( visible=False )

    ax.set_xlim( xrange )
    ax.set_ylim( yrange )
    ax.set_zlim( zrange )
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    ax.view_init(elev, azim, roll)

    tr = uproot.open( f"{file}:events" )

    # define the collision point as (0, 0, 0)
    collision_point = np.array( [0, 0, 0] )

    # beam axis
    em = tr.arrays( ["MCem_px", "MCem_py", "MCem_pz"], entry_start=event_number, entry_stop=event_number+1, library="np" )
    ep = tr.arrays( ["MCep_px", "MCep_py", "MCep_pz"], entry_start=event_number, entry_stop=event_number+1, library="np" )
    em = conv_uproot_to_unit_vec(em, "MCem_p")
    ep = conv_uproot_to_unit_vec(ep, "MCep_p")

    # DRAW THE BEAM

    # ep 
    ep1 = Arrow3D( [ -2*stretch*ep[2], collision_point[2] ],
                   [ -2*stretch*ep[1], collision_point[1] ],
                   [ -2*stretch*ep[0], collision_point[0] ],
                   mutation_scale=12,
                   lw=2, arrowstyle="-", color='k', zorder=0 )
    ep2 = Arrow3D( [ -2*stretch*ep[2], -0.7*stretch*ep[2] ],
                   [ -2*stretch*ep[1], -0.7*stretch*ep[1] ],
                   [ -2*stretch*ep[0], -0.7*stretch*ep[0] ],
                   mutation_scale=12,
                   lw=2, arrowstyle="-|>, head_length=0.7", color='k', zorder=0 )
    ax.add_artist(ep1)
    ax.add_artist(ep2)
    
    # em 
    em1 = Arrow3D( [ -2*stretch*em[2], collision_point[2] ],
                   [ -2*stretch*em[1], collision_point[1] ],
                   [ -2*stretch*em[0], collision_point[0] ],
                   mutation_scale=12,
                   lw=2, arrowstyle="-", color='k', zorder=0 )
    em2 = Arrow3D( [ -2*stretch*em[2], -0.7*stretch*em[2] ],
                   [ -2*stretch*em[1], -0.7*stretch*em[1] ],
                   [ -2*stretch*em[0], -0.7*stretch*em[0] ],
                   mutation_scale=12,
                   lw=2, arrowstyle="-|>, head_length=0.7", color='k', zorder=0 )
    ax.add_artist(em1)
    ax.add_artist(em2)
    
    # DRAW THE THRUST

    # thrust axis
    thrust = tr.arrays( ["EVT_Thrust_x", "EVT_Thrust_y", "EVT_Thrust_z"], entry_start=event_number, entry_stop=event_number+1, library="np" )
    thrust = conv_uproot_to_unit_vec(thrust, "EVT_Thrust_")

    thr1 = Arrow3D( [-1.4*stretch*thrust[2], 1.4*stretch*thrust[2] ],
                    [-1.4*stretch*thrust[1], 1.4*stretch*thrust[1] ],
                    [-1.4*stretch*thrust[0], 1.4*stretch*thrust[0] ],
                    mutation_scale=18,
                    lw=0, arrowstyle='-|>, head_length=1.1, head_width=0.3', color='0.4', clip_on=False )
    thr2 = Arrow3D( [-1.4*stretch*thrust[2], 1.3*stretch*thrust[2] ],
                    [-1.4*stretch*thrust[1], 1.3*stretch*thrust[1] ],
                    [-1.4*stretch*thrust[0], 1.3*stretch*thrust[0] ],
                    mutation_scale=18,
                    lw=1.8, arrowstyle='-', color='0.4', linestyle='dashed', clip_on=False )

    ax.add_artist(thr1)
    ax.add_artist(thr2)
    
    # DRAW THE HEMISPHERE PLANE
    d = collision_point.dot(thrust)
    xx, yy = np.meshgrid( np.linspace(hemis_stretch*xrange[0], hemis_stretch*xrange[1], 20), np.linspace(hemis_stretch*yrange[0], hemis_stretch*yrange[1], 20) )

    if thrust[2]!=0:
        zz = (-thrust[0]*xx - thrust[1]*yy + d) / thrust[2]

        ax.plot_surface(zz, yy, xx, alpha=0.3, color='0.4', label='Hemisphere Defining Plane', zorder=4, clip_on=False)

        x_corners = [xx[0,0], xx[0,-1], xx[-1,-1], xx[-1,0], xx[0,0]]
        y_corners = [yy[0,0], yy[0,-1], yy[-1,-1], yy[-1,0], yy[0,0]]
        z_corners = [zz[0,0], zz[0,-1], zz[-1,-1], zz[-1,0], zz[0,0]]
        ax.plot(z_corners, y_corners, x_corners, color='0.4', lw=0.5, zorder=5, alpha=0.5, clip_on=False)

    
    # PLOT THE B and NU NU
    pv = tr.arrays( ["MCZ_orivtx_x", "MCZ_orivtx_y", "MCZ_orivtx_z"], entry_start=event_number, entry_stop=event_number+1, library="np" )
    pv = conv_uproot_to_unit_vec( pv, 'MCZ_orivtx_', False )

    mc_parts = tr.arrays( ["MC_px", "MC_py", "MC_pz", "MC_q", "MC_orivtx_x", "MC_orivtx_y", "MC_orivtx_z", "MC_PDG", "MC_D1", "MC_D2"], entry_start=event_number, entry_stop=event_number+1 )

    b_parts = mc_parts[ abs(mc_parts['MC_PDG'])==511 ]
    if len(b_parts['MC_PDG'][0])>0:
        d1_ind = b_parts['MC_D1']
        d2_ind = b_parts['MC_D2']
        d1 = mc_parts[d1_ind]['MC_PDG']
        d2 = mc_parts[d2_ind]['MC_PDG']
        b_parts['MC_D1_PDG'] = d1
        b_parts['MC_D2_PDG'] = d2

        b_cand = b_parts[ (abs(b_parts['MC_D1_PDG'])==12) & (abs(b_parts['MC_D2_PDG'])==12) ]
        b_vx, b_vy, b_vz = b_cand['MC_orivtx_x'][0,0], b_cand['MC_orivtx_y'][0,0], b_cand['MC_orivtx_z'][0,0]
        b_px, b_py, b_pz = b_cand['MC_px'][0,0], b_cand['MC_py'][0,0], b_cand['MC_pz'][0,0]
        nu1 = mc_parts[ b_cand['MC_D1'] ]
        nu2 = mc_parts[ b_cand['MC_D2'] ]
        nu1_vx, nu1_vy, nu1_vz = nu1['MC_orivtx_x'][0,0], nu1['MC_orivtx_y'][0,0], nu1['MC_orivtx_z'][0,0]
        nu1_px, nu1_py, nu1_pz = nu1['MC_px'][0,0], nu1['MC_py'][0,0], nu1['MC_pz'][0,0]
        nu2_vx, nu2_vy, nu2_vz = nu2['MC_orivtx_x'][0,0], nu2['MC_orivtx_y'][0,0], nu2['MC_orivtx_z'][0,0]
        nu2_px, nu2_py, nu2_pz = nu2['MC_px'][0,0], nu2['MC_py'][0,0], nu2['MC_pz'][0,0]
        nu1_p = np.sqrt( nu1_px**2 + nu1_py**2 + nu1_pz**2 )
        nu2_p = np.sqrt( nu2_px**2 + nu2_py**2 + nu2_pz**2 )

    
        b = Arrow3D( [b_vz-pv[2], nu1_vz-pv[2]],
                     [b_vy-pv[1], nu1_vy-pv[1]],
                     [b_vx-pv[0], nu1_vx-pv[0]],
                     mutation_scale=18,
                     lw=1.8, arrowstyle='-', color='blue' )
        ax.add_artist(b)

        n1 = Arrow3D( [nu1_vz-pv[2], nu1_vz-pv[2]+0.7*stretch*nu1_pz/nu1_p],
                      [nu1_vy-pv[1], nu1_vy-pv[1]+0.7*stretch*nu1_py/nu1_p],
                      [nu1_vx-pv[0], nu1_vx-pv[0]+0.7*stretch*nu1_px/nu1_p],
                      mutation_scale=18,
                      lw=1.8, arrowstyle='-', color='blue', linestyle=':' )
        ax.add_artist(n1)

        n2 = Arrow3D( [nu2_vz-pv[2], nu2_vz-pv[2]+0.7*stretch*nu2_pz/nu2_p],
                      [nu2_vy-pv[1], nu2_vy-pv[1]+0.7*stretch*nu2_py/nu2_p],
                      [nu2_vx-pv[0], nu2_vx-pv[0]+0.7*stretch*nu2_px/nu2_p],
                      mutation_scale=18,
                      lw=1.8, arrowstyle='-', color='blue', linestyle=':' )
        ax.add_artist(n2)
        
    # PLOT THE TRACKS
    rec_parts = tr.arrays( ["Rec_true_orivtx_x", "Rec_true_orivtx_y", "Rec_true_orivtx_z", "Rec_true_px", "Rec_true_py", "Rec_true_pz", "Rec_true_PDG"], entry_start=event_number, entry_stop=event_number+1 )
    
    # tracks
    tracks = rec_parts[ (abs(rec_parts["Rec_true_PDG"])==2212) | (abs(rec_parts["Rec_true_PDG"])==211) | (abs(rec_parts["Rec_true_PDG"])==321) | (abs(rec_parts["Rec_true_PDG"])==11) | (abs(rec_parts["Rec_true_PDG"])==13) ]
    tracks_vx, tracks_vy, tracks_vz = tracks["Rec_true_orivtx_x"][0], tracks["Rec_true_orivtx_y"][0], tracks["Rec_true_orivtx_z"][0]
    tracks_px, tracks_py, tracks_pz = tracks["Rec_true_px"][0], tracks["Rec_true_py"][0], tracks["Rec_true_pz"][0]
    tracks_id = tracks['Rec_true_PDG'][0]
    
    for vx, vy, vz, px, py, pz, id in zip( tracks_vx, tracks_vy, tracks_vz, tracks_px, tracks_py, tracks_pz, tracks_id):
        
        color = 'mediumseagreen'
        if abs(id) in [11,13]:
            color = 'darkviolet'

        p = np.sqrt( px**2 + py**2 + pz**2 )

        part = Arrow3D( [vz-pv[2], vz-pv[2]+stretch*pz/p],
                        [vy-pv[1], vy-pv[1]+stretch*py/p],
                        [vx-pv[0], vx-pv[0]+stretch*px/p],
                        lw=0.8, arrowstyle='-', color=color )
        ax.add_artist(part)
   
    # DRAW THE VERTICES
    
    # sort the plot first to help with overlays
    if clip is not None:
        fig.subplots_adjust(left=clip[0], right=clip[1], bottom=clip[2], top=clip[3])
    fig.canvas.draw()
    proj = ax.get_proj()

    # draw the PV first
    x, y, z = collision_point[2], collision_point[1], collision_point[0] 
    x2d, y2d, _ = proj3d.proj_transform(x, y, z, proj)
    x_disp, y_disp = ax.transData.transform((x2d, y2d))

    # circ = Circle((x_disp, y_disp),
    #               radius=10,
    #               color='red',
    #               transform=IdentityTransform(),
    #               zorder=100)
    
    radius = 4
    da = DrawingArea(2*radius, 2*radius, 0, 0)
    newcirc = Circle((radius,radius), radius, color='r')
    da.add_artist(newcirc)
    
    ab = AnnotationBbox(da, (x2d, y2d), xycoords='data', frameon=False, box_alignment=(0.5,0.5))
    ax.add_artist(ab)
    ab.set_zorder(10000)

    # fig.patches.append(circ)
    
    # draw the SVs
    rec_vx = rec_parts["Rec_true_orivtx_x"][0]
    rec_vy = rec_parts["Rec_true_orivtx_y"][0]
    rec_vz = rec_parts["Rec_true_orivtx_z"][0]

    for vx, vy, vz in zip(rec_vx, rec_vy, rec_vz):

        dist_to_pv = np.sqrt( (vx-pv[0])**2 + (vy-pv[1])**2 + (vz-pv[2])**2 )
        if np.sqrt( (vx-pv[0])**2 + (vy-pv[1])**2 + (vz-pv[2])**2 ) < 1e-2:
            continue

        x, y, z = vz-pv[2], vy-pv[1], vx-pv[0]
        x2d, y2d, _ = proj3d.proj_transform(x, y, z, proj)
        x_disp, y_disp = ax.transData.transform((x2d, y2d))

        # circ = Circle((x_disp, y_disp),
        #               radius=6,
        #               color='green',
        #               transform=IdentityTransform(),
        #               zorder=100)

        radius = 2
        da = DrawingArea(2*radius, 2*radius, 0, 0)
        newcirc = Circle((radius,radius), radius, color='g')
        da.add_artist(newcirc)
        
        ab = AnnotationBbox(da, (x2d, y2d), xycoords='data', frameon=False, box_alignment=(0.5,0.5))
        ax.add_artist(ab)
        ab.set_zorder(10000)

        # fig.patches.append(circ)
    
    fig.canvas.draw()  # redraw to show overlays

    if save is not None:
        fig.savefig(save)

def make_legend():

    fig = plt.figure(figsize=(6,1))
    ax = fig.add_subplot(111)
    ax.axis('off')

    pv = Line2D([0], [0], marker='o', color='r', label='PV', markerfacecolor='r', markersize=6, lw=0)
    sv = Line2D([0], [0], marker='o', color='g', label='DV', markerfacecolor='g', markersize=3, lw=0)
    thrust = Line2D([0], [0], color='0.4', linestyle='--', label='Thrust Vector', lw=1.2)
    hemis = Patch(color='0.4', alpha=0.5, label='Hemisphere Defining Plane')
    bs = Line2D([0], [0], color='b', lw=1.2, label='$B^{0}$')
    nu = Line2D([0], [0], color='b', lw=1.2, ls=':', label=r'$\nu$')
    lep = Line2D([0], [0], color='darkviolet', lw=0.8, label=r'$\mu^{\pm}$, $e^\pm$')
    had = Line2D([0], [0], color='mediumseagreen', lw=0.8, label=r'$\pi^{\pm}$, $K^\pm$, $p^\pm$')
    
    # legend = Legend(ax, handles=[pv, sv, thrust, hemis, bs, nu, lep, had], ncol=4, loc='center', frameon=False)
    # ax.add_artist(legend)
    ax.legend( handles=[pv, sv, thrust, hemis, bs, nu, lep, had], ncol=4, loc='center' )
    fig.tight_layout()
    fig.savefig("figs/evt_disp_legend.pdf", bbox_inches="tight")

if __name__ == "__main__":

    draw_event_display( os.path.join( path, samples["Bd"] ), event_choice["Bd"], elev=50, azim=90, clip=[-0.08,1.08,-0.1,1.07], save="figs/evt_disp_Bd.pdf" )
    draw_event_display( os.path.join( path, samples["Zbb"] ), event_choice["Zbb"], elev=50, azim=-90, clip=[-0.08,1.08,-0.1,1.07], hemis_stretch=0.8, save="figs/evt_disp_Zbb.pdf" )
    draw_event_display( os.path.join( path, samples["Zud"] ), event_choice["Zud"], elev=50, azim=-90, clip=[-0.08,1.08,-0.1,1.07], save="figs/evt_disp_Zud.pdf" )
    make_legend()

    plt.show()
