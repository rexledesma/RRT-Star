import tkinter as tk
import numpy as np
from functools import partial

class App:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.master = tk.Tk()
        self.master.title("RRT* Implementation")
        self.__setup_app()

    def __setup_app(self):
        self.__setup_canvas(self.width, self.height)
        frame = tk.Frame(self.master)
        frame.pack(side=tk.LEFT)

        # setup top frame of left side
        tf = tk.Frame(frame)
        tf.pack(side=tk.TOP)
        tk.Label(tf, text="<Shift-Left-Click> to change the position of start state.").pack(side=tk.TOP)
        tk.Label(tf, text="<Ctrl-Left-Click> to change the position of goal state.").pack(side=tk.TOP)
        tk.Label(tf, text="<Shift-Control-Left-Click> to drag and create a rectangular obstacle.").pack(side=tk.TOP)

        # setup bottom frame of left side
        bf = tk.Frame(frame)
        bf.pack(side=tk.BOTTOM)
        tk.Scale(bf, from_=30, to=100, orient=tk.HORIZONTAL, length=200,
            label="Radius of Goal State", command=self.__update_goal_radius).pack()
        tk.Button(bf, text="Run RRT for 25 Iterations", command=partial(self.t.run, 25)).pack()

    def __setup_canvas(self, width, height):
        self.canvas = tk.Canvas(self.master, width=width, height=height)
        self.canvas.pack(side=tk.RIGHT)

        xy_s = 0.25*width, height/2 # start state coords
        xy_g = 0.75*width, height/2 # end state coords

        self.start = self.canvas.create_oval(RRTStar.xy2bbox(xy_s, r=10), fill="black")
        self.goal = self.canvas.create_oval(RRTStar.xy2bbox(xy_g, r=30), fill="yellow")

        self.canvas.bind("<Shift-Button-1>", self.__update_start_state)
        self.canvas.bind("<Control-Button-1>", partial(self.__update_coords, self.goal))
        self.canvas.bind("<Shift-Control-Button-1>", self.__start_obstacle)
        self.canvas.bind("<Shift-Control-B1-Motion>", self.__draw_obstacle)

        self.t = RRTStar(self.start, self.goal, self.canvas)

    def __start_obstacle(self, event):
        x, y = event.x, event.y
        self.sxy = event
        self.obstacle = self.canvas.create_rectangle(x, y, x, y, fill="magenta")
        self.t.add_obstacle(self.obstacle)

    def __draw_obstacle(self, event):
        self.canvas.coords(self.obstacle, self.sxy.x, self.sxy.y, event.x, event.y)

    def __update_start_state(self, event):
        self.__update_coords(self.start, event)
        self.t.update_start_state()

    def __update_coords(self, item, event):
        x1, y1, x2, y2 = self.canvas.coords(item)
        dx, dy = event.x - (x1+x2)/2, event.y - (y1+y2)/2
        self.canvas.coords(item, (x1+dx, y1+dy, x2+dx, y2+dy))

    def __update_goal_radius(self, value):
        xy = self.t.item2xy(self.goal)
        self.canvas.coords(self.goal, RRTStar.xy2bbox(xy, r=float(value)))
        self.t.update_goal_radius()

    def __reset(self):
        self.__setup_canvas(self.width, self.height)

    def run(self):
        self.master.mainloop()

class RRTStar:
    def __init__(self, q_init, q_dest, canvas):
        self.q_init      = q_init
        self.q_dest      = q_dest
        self.canvas      = canvas
        self.obstacles   = []                 # obstacle list of item ids
        self.vertices    = np.array([q_init]) # list of item ids of vertices
        self.edges       = {}                 # edge map of item ids
        self.parent      = {q_init: None}     # track parent list to start
        self.update_goal_radius()

        # insert initial tuple of vertex item, and its xy center
        self.vertices_xy = np.array([self.item2xy(q_init)])
        self.c = {q_init: 0} # cost function
        self.min_cost = float('inf')

    def run(self, k=1):
        """Run k iterations of the RRT* planning algorithm"""
        goal_state_xy = self.item2xy(self.q_dest)
        for _ in range(k):
            x_rand = self.sample()
            xy_new_item = self.extend(x_rand)

            if not xy_new_item: continue

            xy_new = self.item2xy(xy_new_item)

            # check if xy_new is in the radius of the goal
            if (np.linalg.norm(goal_state_xy-xy_new) <= self.goal_radius and
                self.c[xy_new_item] < self.min_cost):
                # reset color of all edges/vertices
                self.reset_color()

                # update min cost to goal state
                self.min_cost = self.c[xy_new_item]

                # change color of edges/vertices on shortest path
                v1, v2 = xy_new_item, self.parent[xy_new_item]
                while v1 != self.q_init:
                    edge = self.edges.get(tuple(sorted([v1, v2])))
                    self.canvas.itemconfig(v1, fill="green4")
                    self.canvas.itemconfig(edge, fill="green4", width=5)
                    v1, v2 = v2, self.parent[v2]


    def extend(self, xy, n=10):
        """
        Attempt to extend the tree using a point.
        Returns the new vertex item id if successful, None otherwise.
        """
        n = min(n, len(self.vertices_xy))
        xy_min, xy_min_item = self.nearest(xy, n=1)
        xy_min, xy_min_item = xy_min[0], xy_min_item[0]
        xy_new = self.steer(xy_min, xy)

        if self.obstacle_free(xy_min, xy_new):
            xy_nears, xy_near_items = self.nearest(xy_new, n=n)

            # add xy_new as a vertex
            xy_new_item = self.canvas.create_oval(RRTStar.xy2bbox(xy_new, r=2), fill="black")
            self.vertices_xy = np.append(self.vertices_xy, [xy_new], axis=0)
            self.vertices = np.append(self.vertices, xy_new_item)

            # initialize xy_new's cost as infinity
            self.c[xy_new_item] = float('inf')

            for idx in range(n):
                xy_near = xy_nears[idx,:]
                xy_near_item = xy_near_items[idx]
                if self.obstacle_free(xy_near, xy_new):
                    c_prime = self.c[xy_near_item] + np.linalg.norm(xy_new-xy_near)
                    if c_prime < self.c[xy_new_item]:
                        self.c[xy_new_item] = c_prime
                        xy_min = xy_near
                        xy_min_item = xy_near_item

            # add edge
            x1, y1, x2, y2 = xy_min[0], xy_min[1], xy_new[0], xy_new[1]
            edge = self.canvas.create_line(x1, y1, x2, y2, fill="black")
            self.edges[tuple(sorted([xy_min_item, xy_new_item]))] = edge

            # update parent
            self.parent[xy_new_item] = xy_min_item

            # rewire vertices that can be accessed through xy_new with smaller cost
            for idx in range(n):
                xy_near = xy_nears[idx,:]
                xy_near_item = xy_near_items[idx]
                c_prime = self.c[xy_new_item] + np.linalg.norm(xy_new-xy_near)

                # update cost of edge if smaller cost through xy_new
                if (self.obstacle_free(xy_near, xy_new) and
                    c_prime < self.c[xy_near_item]):
                    self.c[xy_near_item] = c_prime
                    xy_parent_item = self.parent[xy_near_item]

                    # remove edge of x_parent and xy_near

                    edge = self.edges.pop(tuple(sorted((xy_parent_item, xy_near_item))))
                    self.canvas.delete(edge)

                    # add edge with xy_new and xy_near
                    x1, y1, x2, y2 = xy_new[0], xy_new[1], xy_near[0], xy_near[1]
                    edge = self.canvas.create_line(x1, y1, x2, y2, fill="black")
                    self.edges[tuple(sorted([xy_near_item, xy_new_item]))] = edge
                    self.parent[xy_near_item] = xy_new_item

            return xy_new_item

        return None

    def nearest(self, x, n=1):
        """Returns closest n points to x"""
        distances = np.linalg.norm(self.vertices_xy - x, axis=1)
        ind = np.argsort(distances)[:n]

        return self.vertices_xy[ind,:], self.vertices[ind]

    def steer(self, x, y, eta=10):
        """
        Returns a point z that is closer to y than x,
        but is still within eta radius of x
        """
        direction = (y-x)/np.linalg.norm(y-x)
        return x + eta*direction

    def sample(self):
        """Generate random configuration in the space of X_free"""
        self.canvas.update()
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        return np.random.rand(2) * (width, height)

    def obstacle_free(self, a, b, n=50):
        """
        Determine whether line segment from a to b is obstacle free using
        a sampling based method.
        """
        samples = np.random.uniform(low=a, high=b, size=(n, 2))
        return not np.any(np.apply_along_axis(self.is_inside_obstacle, axis=1, arr=samples))

    def is_inside_obstacle(self, a):
        """Checks whether point a is inside any obstacle"""
        for o in self.obstacles:
            x1, y1, x2, y2 = self.canvas.coords(o) # lower/upper coords of rect
            xl, xr = sorted([x1, x2])
            yl, yr = sorted([y1, y2])
            is_inside = xl <= a[0] <= xr and yl <= a[1] <= yr
            if is_inside:
                return True

        return False

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def reset_color(self):
        for v in self.vertices: self.canvas.itemconfig(v, fill="black")
        for e in self.edges.values(): self.canvas.itemconfig(e, fill="black", width=1)

    def update_start_state(self):
        self.vertices_xy[0,:] = self.item2xy(self.q_init)

    def update_goal_radius(self):
        x1, _, x2, _ = self.canvas.coords(self.q_dest)
        mxy = self.item2xy(self.q_dest)
        r = mxy[0] - min(x1, x2)
        self.goal_radius = r

    @staticmethod
    def xy2bbox(xy, r=5):
        """Returns bounding box with xy as center, with radius r"""
        x, y = xy[0], xy[1]
        return x-r, y-r, x+r, y+r

    @staticmethod
    def bbox2xy(bbox):
        """Returns xy as center, given bounding box"""
        x1, y1, x2, y2 = bbox
        return np.array(((x1+x2)/2, (y1+y2)/2))

    def item2xy(self, item):
        """Returns xy as center, given item id"""
        return self.bbox2xy(self.canvas.coords(item))

if __name__ == "__main__":
    App().run()