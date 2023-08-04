import numpy as np
import matplotlib.pyplot as plt

class Triangulator:
    def __init__(self, labels, decay_type, electrode_dict, neuron_dict, grid_size) -> None:
        self.labels = labels
        self.decay_type = decay_type
        self.electrode_dict = electrode_dict
        self.neuron_dict = neuron_dict
        self.grid_size = grid_size

    def __draw_circle(self, plot, point, radius):
        x, y = point[0], point[1]
        return plot.Circle((x, y), radius, fill=False, edgecolor='grey')
    
    def __signal_strength_ratio(self, signal_1, signal_2):
        if self.decay_type == "square":
            ratio = np.sqrt(signal_1 / signal_2)
        else:
            print("Unknown Decay Type:", self.decay_type)

        return ratio
    
    def __is_point_on_line(self, point, line_point1, line_point2):
        """ Check if a point is on a line between two other points """
        
        # get the coordinates of the points
        x, y = point
        x1, y1 = line_point1
        x2, y2 = line_point2

        # compute the cross product and the dot product
        cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
        dot_product = (x - x1) * (x2 - x1) + (y - y1)*(y2 - y1)

        # compute the squared length of the line segment
        squared_length = (x2 - x1)**2 + (y2 - y1)**2

        # check if the point is on the line segment
        if abs(cross_product) > 1e-10 or dot_product < 0 or dot_product > squared_length:
            return False
        else:
            return True

    def __find_intersection(self, lines):
        line1, line2 = lines[0], lines[1]

        m1, b1 = line1
        m2, b2 = line2

        if m1 == m2:
            print('Lines are parallel and do not intersect.')
            return None
        
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y
    
    def __calculate_straight_line(self, point_1, point_2):
        x_1, y_1, x_2, y_2 = point_1[0], point_1[1], point_2[0], point_2[1]

        if x_2 - x_1 == 0:
            return float('inf'), x_1  # return 'inf' as m and x_1 as 'b' for vertical line
        elif y_2 - y_1 == 0:
            m = 0
            b = y_1
        else:
            m = (y_2 - y_1) / (x_2 - x_1)
            b = y_1 - (m * x_1)

        return m, b

    def __calculate_perpendicular(self, point, m_original, b_original):
        x, y = point

        # handle the case of the original line being vertical
        if m_original == float('inf'):
            m_perp = 0
            b_perp = y

        # handle the case of the original line being horizontal
        elif m_original == 0:
            m_perp = float('inf')
            b_perp = x

        # for all other lines
        else:
            m_perp = -1 / m_original
            b_perp = y - m_perp * x

        return m_perp, b_perp

    def __calculate_tangent(self, point_1, point_2, radius):
        x_c, y_c = point_1[0], point_1[1]

        # gradient and intercept of line
        m, b = self.__calculate_straight_line(point_1, point_2)

        # coefficients of the quadratic equation
        A = 1 + m**2
        B = 2 * (m * (b - y_c) - x_c)
        C = x_c**2 + (b - y_c)**2 - radius**2

        # calculate the discriminant
        D = B**2 - 4 * A * C

        # list to store the intersections of circle and line
        intersections = []

        # no intersection (shouldn't happen)
        if D < 0:
            print('The line and the circle do not intersect.')
            return None
        # the line is tangent to the circle
        elif D == 0:  
            x = -B / (2 * A)
            y = m * x + b
            intersections.append((x, y))
        # the line intersects the circle at two points
        else:  
            x1 = (-B + D**0.5) / (2 * A)
            y1 = m * x1 + b
            intersections.append((x1, y1))
            x2 = (-B - D**0.5) / (2 * A)
            y2 = m * x2 + b
            intersections.append((x2, y2))
        
        for intersection in intersections:
            if self.__is_point_on_line(intersection, point_1, point_2):
                return self.__calculate_perpendicular(intersection, m, b)

    def __euclidean_distance(self, point_1, point_2):
        return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

    def show_grid(self, show_construction=True) -> None:
        for neuron_id in range(len(np.unique(self.labels))):
            fig, ax = plt.subplots(figsize=(6,6))
            
            # extract neuron and electrode positions
            neuron_positions = [(v["row"], v["col"]) for v in self.neuron_dict.values()]
            electrode_positions = [(v["row"], v["col"]) for v in self.electrode_dict.values()]

            # if there are neurons, plot them
            if neuron_positions:
                neuron_rows, neuron_cols = zip(*neuron_positions) # unzip into x and y coordinates
                plt.scatter(neuron_rows, neuron_cols, color='r', label='Neurons', s=100)

                # add text labels for neurons
                for i, (x, y) in enumerate(neuron_positions):
                    plt.text(x, y, f'{i}', fontsize=20)

            # if there are electrodes, plot them
            if electrode_positions:
                electrode_rows, electrode_cols = zip(*electrode_positions) # unzip into x and y coordinates
                plt.scatter(electrode_rows, electrode_cols, color='b', label='Electrodes', s=100)

                # add text labels for electrodes
                for i, (x, y) in enumerate(electrode_positions):
                    plt.text(x, y, f'{i}', fontsize=20)

            # draw lines for each pair of points
            for pair in self.triangulate_info[neuron_id]["electrode_pairs"]:
                point1, point2 = pair
                plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-')

            # show construction lines and circles 
            if show_construction:
                # plot circles
                for info in self.triangulate_info[neuron_id]["circle_info"]:
                    circle = self.__draw_circle(plt, info[0], info[1])
                    ax.add_patch(circle)
                
                # plot the perpendicular lines
                for slope, intercept in self.triangulate_info[neuron_id]["perpendicular_lines"]:
                    x = np.linspace(0, self.grid_size, 1000)
                    y = slope * x + intercept
                    plt.plot(x, y, c="blue", lw=0.5)

            # plot the intersection point
            intersection_coords = self.triangulate_info[neuron_id]["intersection_point"]
            plt.scatter(intersection_coords[0], intersection_coords[1], c="purple", s=100, label="Estimated Position")

            plt.xlim(0, self.grid_size - 1)
            plt.ylim(0, self.grid_size - 1)
            plt.grid(True)
            plt.legend()
            plt.title(f"Identifying Neuron {neuron_id} Location")
            plt.show()
    
    def triangulate(self):
        self.triangulate_info = {}
        # iterate through each of the identified neurons
        for neuron_id in range(len(np.unique(self.labels))):
            self.triangulate_neuron(neuron_id)

    def triangulate_neuron(self, neuron_id):
        self.triangulate_info[neuron_id] = {
            "electrode_pairs": [],
            "perpendicular_lines": [],
            "circle_info": [],
        }

        # get the items of the electrode dictionary as a list of tuples
        items = list(self.electrode_dict.items())

        # sort the electrodes by the strength of the signal recorded
        sorted_items = sorted(items, key=lambda item: item[1]["avg_waveform_peak"][neuron_id])

        # convert the sorted items back into a dictionary.
        sorted_dict = dict(sorted_items)

        # ids for the electrodes in order of signal strength 
        electrodes_by_strength = list(sorted_dict.keys())[::-1]
        self.triangulate_info[neuron_id]["electrodes_by_strength"] = electrodes_by_strength

        # distance between first electrode (a) and second electrode (b)
        x_a, y_a = self.electrode_dict[electrodes_by_strength[0]]["row"], self.electrode_dict[electrodes_by_strength[0]]["col"]
        x_b, y_b = self.electrode_dict[electrodes_by_strength[1]]["row"], self.electrode_dict[electrodes_by_strength[1]]["col"]

        d_ab = self.__euclidean_distance((x_a, y_a), (x_b, y_b))
        self.triangulate_info[neuron_id]["electrode_pairs"].append(((x_a, y_a), (x_b, y_b)))

        # ratio of signal strength between the electrodes a and b
        s_a =  self.electrode_dict[electrodes_by_strength[0]]["baseline_to_peak"][neuron_id]
        s_b =  self.electrode_dict[electrodes_by_strength[1]]["baseline_to_peak"][neuron_id]

        ab_ratio = self.__signal_strength_ratio(s_a, s_b)


        # neurons distance from electrode a along the line connecting a and b
        r_ab = d_ab / (1 + ab_ratio)
        self.triangulate_info[neuron_id]["perpendicular_lines"].append(self.__calculate_tangent((x_a, y_a), (x_b, y_b), r_ab))
        self.triangulate_info[neuron_id]["circle_info"].append(((x_a, y_a), r_ab))

        # distance between first electrode (a) and third electrode (c)
        x_c, y_c = self.electrode_dict[electrodes_by_strength[2]]["row"], self.electrode_dict[electrodes_by_strength[2]]["col"]
        d_ac = np.sqrt((x_a - x_c)**2 + (y_a - y_c)**2)
        self.triangulate_info[neuron_id]["electrode_pairs"].append(((x_a, y_a), (x_c, y_c)))

        # ratio of signal strength between the electrodes a and b
        s_c =  self.electrode_dict[electrodes_by_strength[2]]["baseline_to_peak"][neuron_id]

        ac_ratio = self.__signal_strength_ratio(s_a, s_c)

        # neurons distance from electrode a along the line connecting a and b
        r_ac = d_ac / (1 + ac_ratio)
        self.triangulate_info[neuron_id]["perpendicular_lines"].append(self.__calculate_tangent((x_a, y_a), (x_c, y_c), r_ac))
        self.triangulate_info[neuron_id]["circle_info"].append(((x_a, y_a), r_ac))

        self.triangulate_info[neuron_id]["intersection_point"] = self.__find_intersection(self.triangulate_info[neuron_id]["perpendicular_lines"])

    def assess_triangulation(self, show_results=False):
        # iterate through each of the identified neurons
        for neuron_id in range(len(np.unique(self.labels))):

            # get the coordinates of the predicited neuron location
            calculated_position = self.triangulate_info[neuron_id]["intersection_point"]
            x_calc, y_calc = calculated_position[0], calculated_position[1]

            # get true location of the neuron
            x_true, y_true = self.neurons_dict[neuron_id]["row"], self.neurons_dict[neuron_id]["col"]

            # get distance between the two
            distance = self.__euclidean_distance((x_true, y_true), (x_calc, y_calc))

            # add to the dictionary
            self.triangulate_info[neuron_id]["distance_from_true_position"] = distance

            if show_results:
                print(f"Estimated position of neuron {neuron_id} is {distance} from the true position")