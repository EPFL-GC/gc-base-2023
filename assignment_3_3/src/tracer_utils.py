import igl
import numpy as np
import math
import numpy.linalg as la
from tracer_helper import rotate_vector

def asymptotic_path(idx, mesh, num_steps, step_size, first_principal_direction, num_neighbors, sampling_dist=0):
    '''Computes both tracing direction (backward and forward) following an asymptotic path.

    Inputs:
    - idx : int
        The index of the vertex on the mesh to start tracing.
    - mesh : Mesh
        The mesh for tracing.
    - num_steps : int
        The number of tracing steps.
    - step_size : int
        The size of the projection at each tracing step.
    - first_principal_direction : bool
        Indicator for using the first principal curvature to do the tracing.
    - num_neighbors : int
        Number of closest vertices to consider for avering principal curvatures.
    - sampling_distance : float
        The distance to sample points on the path (For the design interface - Don't need to be define).
            
    Outputs:
    - P : np.array (n, 3)
        The ordered set of unique points representing a full asymptotic path.
    - A : np.array (n,)
        The ordered set of deviated angles in degrees calculated for the asymptotic directions.
    - PP : np.array (m,3)
        The ordered set of points laying on the path spaced with a given distance (For the design interface).
    '''

    path_forwards, angles_forwards, samples_forwards = trace(idx, mesh, num_steps, step_size, first_principal_direction, False, num_neighbors, sampling_dist)
    path_backwards, angles_backwards, samples_backwards = trace(idx, mesh, num_steps, step_size, first_principal_direction, True, num_neighbors, sampling_dist)

    # Remove duplicate point
    P = path_forwards[1:]
    A = angles_forwards[1:]
    # Reverse array
    P = P[::-1]
    A = A[::-1]
    PP = samples_forwards[::-1]
        
    #Combine arrays
    P = np.append(P, path_backwards, axis=0)
    A = np.append(A, angles_backwards, axis=0)
    PP = np.append(PP, samples_backwards, axis=0)

    return P, A, PP

def trace(idx, mesh, num_steps, step_size, first_principal_direction, trace_backwards, num_neighbors, sampling_dist=0):
    '''Computes one tracing direction following an asymptotic path.

    Inputs:
    - idx : int
        The index of the vertex on the mesh to start tracing.
    - mesh : Mesh
        The mesh for tracing.
    - num_steps : int
        The number of tracing steps.
    - step_size : int
        The size of the projection at each tracing step.
    - first_principal_direction : bool
        Indicator for using the first principal curvature to do the tracing.
    - trace_backwards : bool
        Indicator for mirroring the deviated angle
    - num_neighbors : int
        Number of closest vertices to consider for avering principal curvatures.
    - sampling_distance : float
        The distance to sample points on the path (For the design interface - Don't need to be define).
            
    Outputs:
    - P : np.array (n, 3)
        The ordered set of points representing one tracing direction.
    - A : np.array (n,)
        The ordered set of deviated angles calculated for the asymptotic directions.
    - PP : np.array (m,3)
        The ordered set of points laying on the path spaced with a given distance (For the design interface).
    '''

    P = np.empty((0, 3), float)
    PP = np.empty((0,3), float)
    A = np.array([],float)

    #Get the data of the first vertex in the path
    pt = mesh.V[idx]

    # Store partial distance
    partial_dist = 0
    while len(P) < num_steps:

        # Add the current point to the path
        P = np.append(P, np.array([pt]), axis=0)

        # Get the averaged principal curvature directions & values
        k1_aver, k2_aver, v1_aver, v2_aver, n_aver = averaged_principal_curvatures(pt, mesh, num_neighbors)

        # Calculate deviation angle (theta) based on principal curvature values
        theta = 2*math.atan(math.sqrt((2 * math.sqrt(k2_aver * (k2_aver - k1_aver)) + k1_aver - 2 * k2_aver) / k1_aver))
            
        #Store theta
        A = np.append(A, np.array([theta]), axis=0)

        # Mirror the angle for tracing backwards. Use trace_backwards indicator
        if trace_backwards:
            theta += np.pi

        # Rotate principal curvature direction to get asymptotic direction. Use first_principal_direction indicator
        if first_principal_direction:
            a_dir = rotate_vector(v1_aver, theta, v1_aver, v2_aver, n_aver)
        else:
            a_dir = rotate_vector(v1_aver, -theta, v1_aver, v2_aver, n_aver)

        # Check for anticlastic surface-regions
        if k1_aver * k2_aver > 0:
            break

        # Check for valid asymptotic direction and unitize
        if la.norm(a_dir) == 0:
            break
        else:
            a_dir /= la.norm(a_dir)

        # Prevent the tracer to go in the opposite direction
        N = len(P)
        if N > 1:
            a_dir *= np.sign(np.dot(a_dir, P[N-1] - P[N-2]))

        # Scale the asymptotic direction to the given step-size
        a_dir *= step_size

        # Compute edge-point
        edge_point, is_boundary_edge = find_edge_point(mesh, pt, a_dir)

        # Check for boundaries
        if is_boundary_edge:
            if N > 1:
                P = np.append(P, np.array([edge_point]), axis=0)
            break

        #Check for duplicated points
        dist = la.norm(edge_point-pt)
        if dist < 1e-6:
            break

        # Store sampling points (For the design interface)
        if sampling_dist>0:
            partial_dist += dist
            if partial_dist >= sampling_dist :
                partial_dist = 0
                PP = np.append(PP, np.array([edge_point]), axis=0)

        pt = edge_point

    return P, A, PP
    
def averaged_principal_curvatures(pt, mesh, num_neighbors=2, eps=1e-6):
    '''Computes inverse weighted distance average of principal curvatures of a given mesh-point
       on the basis of the two closest vertices.
    Try to compute values, directions and normal at the given query point.

    Inputs:
    - pt : np.array (3,)
        The query point position.
    - mesh : Mesh
        The mesh for searching nearest vertices.
    - num_neighbors : int
        Number of closest vertices to consider for avering.
    - eps : float
        The distance tolerance to consider whether the given point and a mesh-vertex are coincident.
            
    Outputs:
    - k_1 : float
        The min principal curvature average at the given query point.
    - k_2 : float
        The max principal curvature average at the given query point.
    - v1_aver : np.array (3,)
        The unitized min principal curvature direction average at the given query point.
    - v2_aver : np.array (3,)
        The unitized max principal curvature direction average at the given query point.
    - n_aver : np.array (3,)
        The unitized normal average at the given query point.
    '''

    # Get the closest vertices and distances to the query point
    # Use these data to compute principal curvature weighted averages.
    dist, neighbors = mesh.get_closest_vertices(pt, num_neighbors)

    v1_aver = None
    v2_aver = None
    k1_aver = None
    k2_aver = None
    n_aver = None

    return k1_aver, k2_aver, v1_aver, v2_aver, n_aver

def find_edge_point(mesh, a_orig, a_dir):
    '''Computes the point where a mesh-edge intersects with the asymptotic projection.
    Try to compute the edge-point resulting from this intersection.

    Inputs:
    - mesh : Mesh
        The mesh for searching edge intersections.
    - a_orig : np.array (3,)
        The start position of the asymptotic projection.
    - a_dic : np.array (3,)
        The direction of the asymptotic projection.
            
    Outputs:
    - edge_point : np.array (3,)
        The position of the edge-point.
    - is_boundary_point : bool
        Indicator for whether the edge-point is at the boundary of the mesh.
    '''

    # Get the closest face-index and mesh-point (point laying on the mesh)
    proj_pt = a_orig+a_dir
    face_index, mesh_point = mesh.get_closest_mesh_point(proj_pt)

    # Update the projection vector with the position of the mesh-point
    a_dir = mesh_point - a_orig

    # If the mesh-point is equal to the starting point, return flag for boundary vertex.
    if la.norm(a_dir)==0:
        return mesh_point, True

    # Unitize projection vector
    a_dir /= la.norm(a_dir)

    # Initialize variables
    edge_point = mesh_point
    is_boundary_point = False
    prev_projection_param = 0
 
    # Find the required edge-point by computing intersections between the edge-segments of the face and the asymptotic-segment. 
    # Different intersection events need to be considered. 
    # You can call the function <edge_faces> from the mesh class for checking bounaries. 
    edges = mesh.face_edges[face_index]
    for e_idx in edges:

        e = mesh.edge_vertices[e_idx]
        e_orig = mesh.V[e[0]]
        e_dir = mesh.V[e[1]]-e_orig
        is_boundary_edge = np.any(mesh.edge_faces[e_idx]== -1)

        edge_param, projection_param, intersection = intersection_event(e_orig, e_dir, a_orig, a_dir)

        # TODO: Find the edge-point

    return edge_point, is_boundary_point

def intersection_event(a_orig, a_dir, b_orig, b_dir, eps=1e-6):
    '''Computes the intersection event between segments A and B.
    Try to compute the intersection event.

    Inputs:
    - a_orig : np.array (3,)
        The start position of segment A.
    - a_dic : np.array (3,)
        The direction of the segment A.
    - b_orig : np.array (3,)
        The start position of segment B.
    - b_dic : np.array (3,)
        The direction of the segment B.
    - eps : float
        The tolerance for determining intersections.
            
    Outputs:
    - t : float
        The parameter on segment A where the intersection occurred.
    - u : float
        The parameter on segment B where the intersection occurred.
    - E  : int
        Indicator for the type of intersection event. 
        Returns 0 for all intersection events. 
        Returns 1 for collinearity. When E = 1, return None, None for t, u
    '''
    u = None
    t = None
    E = None

    return t, u, E
