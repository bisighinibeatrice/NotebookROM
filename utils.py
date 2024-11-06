import pyvista as pv
import numpy as np

# Check if running in a headless environment
import os
if os.environ.get("DISPLAY") is None:
    pv.start_xvfb()

class Visualizer:
    def __init__(self):
        """
        Visualizer class to plot the stent and vessel in 3D
        """
        self.plotter = pv.Plotter(notebook=True)
        self.plotter.add_axes()
        self.plotter.view_zx()

    def add_stent_from_file(self, point_file, connectivity_file, radius=0.02):
        """
        Add stent mesh from file
        
                Parameters:
                    point_file (str): Path to the point cloud file
                    connectivity_file (str): Path to the connectivity file
                    radius (float): Radius of the stent tube

        """
        self.stent_radius = radius

        # Load point cloud data and connectivities
        points = np.loadtxt(point_file)
        connectivity = np.loadtxt(connectivity_file).astype(np.int32) 
        connectivity = connectivity - np.min(connectivity)  # Ensure that the connectivity starts from 0

        # We must add a column to the connectivities to indicate to vtk how many points per edge
        connectivity = np.hstack((np.ones((connectivity.shape[0], 1), dtype=np.int32)*2, connectivity))

        # Create a PolyData object
        self.stent_mesh = pv.PolyData(points, lines=connectivity)
        self.stent_tube_mesh = self.stent_mesh.tube(radius=radius)

        # Plot the mesh
        self.plotter.add_mesh(self.stent_tube_mesh, color="black")

        self.plotter.view_zx()

    def update_stent(self, new_points):
        """
        Update the stent mesh with new points
            
                Parameters:
                    new_points (np.ndarray): New point cloud data
        """

        assert new_points.shape == self.stent_mesh.points.shape

        self.stent_mesh.points = new_points
        self.stent_tube_mesh.points = self.stent_mesh.tube(radius=self.stent_radius).points

        self.plotter.view_zx()

    def update_stent_from_file(self, point_file):
        """
        Update the stent mesh with new points from file
        
                Parameters:
                    point_file (str): Path to the new point cloud file
        """
        new_points = np.loadtxt(point_file)
        self.update_stent(new_points)

    def add_vessel(self, control_point=np.array([0, 0, 25], dtype=np.float64), vessel_radius=10, deploy_site = 0.5, vessel_opacity=0.5):
        """
        Add vessel centerline, tube, control point and endpoints.
        The vessel is defined by a spline interpolation between the start point, control point and end point.
        The control point is shown in red, the endpoints in blue and the deployment site in green.
        
                Parameters:
                    control_point (np.ndarray, optional): Control point of the vessel. Default is [0, 0, 25]
                    vessel_radius (float, optional): Radius of the vessel tube. Default is 10
                    deploy_site (float, optional): Deployment site of the stent in the vessel. Default is 0.5
                    vessel_opacity (float, optional): Opacity of the vessel tube. Default is 0.5
        """

        # Calculate the spline points and connectivity
        self.start_point = np.array([0, 0, 0], dtype=np.float64)
        self.end_point = np.array([0, 0, 50], dtype=np.float64)
        num_divisions = 100
        points = np.array([spline_interpolation(self.start_point, control_point, self.end_point, u) for u in np.arange(0, 1 + (1 / num_divisions), 1 / num_divisions)])
        connectivity = np.arange(0, len(points), dtype=np.int32)
        connectivity = np.insert(connectivity, 0, len(points))

        # Create centerline and tube
        self.vessel_centerline = pv.PolyData(points, lines=connectivity)
        self.vessel_tube = self.vessel_centerline.tube(radius=vessel_radius, n_sides=40)

        # Create control point and endpoints
        self.vessel_controlpoint = pv.wrap(control_point.reshape(1, 3))
        vessel_endpoints = pv.wrap(np.array([self.start_point, self.end_point]))

        # Create deploy site point
        deploy_site = spline_interpolation(self.start_point, control_point, self.end_point, deploy_site)
        self.deploy_site = pv.wrap(deploy_site.reshape(1, 3))

        # Plot the meshes
        self.plotter.add_mesh(self.vessel_centerline, color="black")
        self.plotter.add_mesh(self.vessel_tube, color="pink", opacity=vessel_opacity)
        self.plotter.add_mesh(self.vessel_controlpoint, color="red", render_points_as_spheres=True, point_size=10)
        self.plotter.add_mesh(vessel_endpoints, color="blue", render_points_as_spheres=True, point_size=5)
        self.plotter.add_mesh(self.deploy_site, color="green", render_points_as_spheres=True, point_size=20)

        self.plotter.view_zx()

    def update_vessel(self, new_control_point, new_radius, new_deploy_site):
        """
        Update the vessel mesh with new control point, radius and deployment site

                Parameters:
                    new_control_point (np.ndarray): New control point of the vessel
                    new_radius (float): New radius of the vessel tube
                    new_deploy_site (float): New deployment site of the stent in the vessel
        """

        self.vessel_centerline.points = np.array([spline_interpolation(self.start_point, new_control_point, self.end_point, u) for u in np.linspace(0, 1, 100)])
        self.vessel_tube.points = self.vessel_centerline.tube(radius=new_radius, n_sides=40).points
        self.vessel_controlpoint.points = new_control_point.reshape(1, 3)
        self.deploy_site.points = spline_interpolation(self.start_point, new_control_point, self.end_point, new_deploy_site).reshape(1, 3)

        self.plotter.view_zx()


    def show(self):
        """
        Show the plot
        """
        return self.plotter.show(jupyter_backend="html")

def binomial_coefficient(n, k):
    """
    Compute the binomial coefficient
    
            Parameters:
                n (int): Total number of elements
                k (int): Number of elements to choose
            
            Returns:
                int: Binomial coefficient
    """
    if k == 0:
        return 1
    else:
        return round(n/k * binomial_coefficient(n-1, k-1))

def spline_interpolation(start_point, control_point, end_point, u):
    
    """
    Compute the spline interpolation between three points
    
            Parameters:
                start_point (np.ndarray): Starting point
                control_point (np.ndarray): Control point
                end_point (np.ndarray): End point
                u (float): Parameter between 0 and 1
                
            Returns:
                np.ndarray: Interpolated point
    """
    point_sum = np.array([0.0, 0.0, 0.0])  # Sum of control point contributions
    control_points = np.array([start_point, control_point, end_point])  # Array of control points
    degree = len(control_points) # Degree of the Bernstein polynomial

    for i in range(1,degree+1):
        # Compute Bernstein basis polynomial B_i^n(u)
        bernstein_value = binomial_coefficient(degree, (i-1)) * (u ** (i-1)) * ((1 - u) ** (degree - (i-1)))
        point_sum += bernstein_value * control_points[(i-1)]  # Sum the control point contributions
    bernstein_value = binomial_coefficient(degree, degree) * (u ** degree)    
    point_sum += bernstein_value * control_points[degree - 1]
  
    return point_sum

def read_input_from_file(filename):
    """
    Reads vessel radius and control points from a file for multiple models.

    The file should contain lines with the format:
    vessel_radius control_point_x control_point_z

    Parameters:
    filename (str): The path to the input file.

    Returns:
    list of tuples: Each tuple contains:
        - vessel_radius (float): The radius of the vessel.
        - control_point (np.array): A 3D numpy array [x, y, z] where y is set to 0 by default.
    
    Example:
    If the file contains:
    1.5 2.0 3.0
    2.0 3.5 4.5
    
    The function will return:
    [(1.5, array([2.0, 0.0, 3.0])), (2.0, array([3.5, 0.0, 4.5]))]
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Parse each line into a dictionary of parameters for each model
    model_params = []
    for line in lines:
        values = line.split()  # Assuming space-separated or comma-separated values
        vessel_radius = float(values[0])
        # The control point can have positions on both X and Z, Y is kept at 0
        control_point = np.array([float(values[1]), 0.0, float(values[2])])
        model_params.append((vessel_radius, control_point))

    return model_params
