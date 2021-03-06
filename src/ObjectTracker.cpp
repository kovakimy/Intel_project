#include "ObjectTracker.hpp"


// functions

float cosineSimilarity(std::vector<float>& A, std::vector<float>& B) {
	size_t size = A.size();
	float res = 0;
	float sumA2 = 0;
	float sumB2 = 0;
	float sumAB = 0;
	for (size_t i = 0; i < size; ++i) {
		sumAB += A[i] * B[i];
		sumA2 += A[i] * A[i];
		sumB2 += B[i] * B[i];
	}
	res = sumAB / (sqrt(sumA2) * sqrt(sumB2));
	return res;
};

template<class T>
vector<T> HungarianAlgorithm(const vector<vector<T>>& g)
{
	int n = g.size() - 1;
	vector<int> par(n + 1, 0);
	vector<int> way(n + 1, 0);
	vector<T> u(n + 1, 0);
	vector<T> v(n + 1, 0);
	for (int i = 1; i < n + 1; i++)
	{
		par[0] = i;
		int prev_col = 0;
		int next_col;
		vector<T> minv(n + 1, INT_MAX);
		vector<bool>  used(n + 1, false);

		do {
			used[prev_col] = true;
			int new_row = par[prev_col];
			T delta = T(INT_MAX);
			for (int j = 1; j < n + 1; j++) {
				if (!used[j])
				{
					T current = g[new_row][j] - u[new_row] - v[j];
					if (current < minv[j]) {
						minv[j] = current;
						way[j] = prev_col;
					}
					if (minv[j] < delta) {
						delta = minv[j];
						next_col = j;
					}
				}
			}
			for (int j = 0; j < n + 1; j++)
			{
				if (used[j]) {
					u[par[j]] += delta;
					v[j] -= delta;
				}
				else
					minv[j] -= delta;
			}
			prev_col = next_col;
		} while (par[prev_col] != 0);

		do
		{
			int j1 = way[prev_col];
			par[prev_col] = par[j1];
			prev_col = j1;
		} while (prev_col != 0);
	}
	v[0] *= -1; // total res
	vector<T> res(n + 1, 0);
	for (int j = 1; j <= n; ++j)
		res[par[j]] = j;
	return res;
}

// =========== class ObjectTracker =========== 

ObjectTracker::ObjectTracker(float not_found_segment_cost,
	float not_found_object_cost) {
	E_t = not_found_segment_cost;
	E_s = not_found_object_cost;
}

/*vector<int> ObjectTracker::SetStartObjects(vector<Object> objects) {
	current_objects = objects;
	vector<int> objects_ids;
	for (auto& item : current_objects) {
		objects_ids.push_back(next_id);
	}
	return objects_ids;
}*/

vector<Object> ObjectTracker::Track(vector<Object> &segments){//(vector<pair<Point, Point>> &segments) {
	Predict();
	vector<Point> segments_centers;
	for (auto& seg : segments)

	/*{
		double x = (seg.pos[0].x + seg.pos[1].x) / 2;
		double y = (seg.pos[0].y + seg.pos[1].y) / 2;
		segments_centers.push_back(Point(x, y));
	}*/

	int max_item = 0;
	int size = current_objects.size() + segments_centers.size();
	int item = 0;
	vector<vector<float>> matrix(size + 1, vector<float>(size + 1));
	vector<float> combination(size + 1);

	// Creating matrix for assignment algorithm
	for (int i = 1; i <= current_objects.size(); i++) {
		for (int j = 1; j <= segments_centers.size(); j++) {
			item = cosineSimilarity(current_objects[i].feature, segments[j].feature);
			max_item = max(max_item, item);
			matrix[i][j] = item;
		}
	}
	max_item *= 10;

	for (int i = 1; i <= current_objects.size(); i++) {
		for (int j = segments_centers.size() + 1; j <= size; j++) {
			if (i == j) {
				matrix[i][j] = E_t;
			}
			else {
				matrix[i][j] = max_item;
			}
		}
	}

	for (int i = current_objects.size() + 1; i <= size; i++) {
		for (int j = 1; j <= segments_centers.size(); j++) {
			if (i == j) {
				matrix[i][j] = E_s;
			}
			else {
				matrix[i][j] = max_item;
			}
		}
	}

	for (int i = current_objects.size() + 1; i < size; i++) {
		for (int j = segments_centers.size() + 1; j < size; j++) {
			matrix[i][j] = 0;
		}
	}

	vector<int> objects_to_del;
	combination = HungarianAlgorithm(matrix);

	for (int i = 0; i < combination.size(); ++i)
	{
		float objID = combination[i], segID = i;

		// if found object for segment
		if (get_Euclidean_dist(current_objects[objID], segments[segID]) < similarityThreshold)
		{
			prev_objects[objID] = current_objects[objID];
			current_objects[objID] = segments[segID];
			current_objects[objID].id = prev_objects[objID].id;
			current_objects[objID].trajectory.push_back(segments_centers[segID]);

			/*
			current_objects[objID].pos[0].x = segments[segID].pos[0].x;
			current_objects[objID].pos[0].y = segments[segID].pos[0].y;
			current_objects[objID].pos[1].x = segments[segID].pos[1].x;
			current_objects[objID].pos[1].y = segments[segID].pos[1].y;
			*/
		}
		
		// if not found any segment for ñurrent object :
		else if (matrix[objID][segID] == E_t)
		{
			if (current_objects[objID].time > (std::chrono::steady_clock::now() + 400ms))
			{
				objects_to_del.push_back(objID);
				break;
			}
		}
		
		// if not found any segment for ñurrent segment (new object on the picture):
		else if (matrix[objID][segID] == E_s)
		{
			Object new_obj = Object(segments[segID].pos, segments[segID].feature, next_id);
			next_id++;
			current_objects.push_back(new_obj);
			//segments_centers[segID].obj = &current_objects.back();
		}
	}

	for (auto& ind : objects_to_del)
	{
		current_objects.erase(current_objects.begin() + ind);
	}

}
