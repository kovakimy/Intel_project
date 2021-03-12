﻿#include "../include/ObjectTracker.hpp"

using namespace std;
using namespace cv;

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
vector<int> HungarianAlgorithm(vector<vector<T>> g)
{
	int n = g.size() - 1;
	vector<int> par(n + 1, 0);
	vector<int> way(n + 1, 0);
	vector<T> u(n + 1, 0);
	vector<T> v(n + 1, 0);

	for (int i = 1; i <= n; ++i)
	{
		T max_in_row = 0;
		for (int j = 1; j <= n; ++j)
		{
			if (max_in_row < g[i][j])
				max_in_row = g[i][j];
		}
		for (int j = 1; j <= n; ++j)
		{
			g[i][j] -= max_in_row;
			g[i][j] *= -1;
		}

	}

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
	vector<int> res(n + 1, 0);
	for (int j = 1; j <= n; ++j)
		res[par[j]] = j;
	return res;
}

// =========== class ObjectTracker =========== 

ObjectTracker::ObjectTracker(float not_found_segment_cost,
	float not_found_object_cost) :
	E_t(not_found_segment_cost / 2.), E_s(not_found_object_cost / 2.) {

}









vector<Object> ObjectTracker::Track(vector<Object>& segments) {//(vector<pair<Point, Point>> &segments) {
	vector<Point> segments_centers;
	/*
	for (auto& seg : segments)
	{
		double x = (seg.pos[0].x + seg.pos[1].x) / 2;
		double y = (seg.pos[0].y + seg.pos[1].y) / 2;
		segments_centers.push_back(Point(x, y));
	}*/

	float max_item = 0;
	int size = current_objects.size() + segments.size();
	float item = 0;
	vector<vector<float>> matrix(size + 1, vector<float>(size + 1));
	vector<int> combination(size + 1);
	// Creating matrix for assignment algorithm
	for (int i = 1; i <= current_objects.size(); i++) {
		for (int j = 1; j <= segments.size(); j++) {
			item = cosineSimilarity(current_objects[i-1].feature, segments[j-1].feature);
			max_item = max(max_item, item);
			matrix[i][j] = item;
		}
	}
	max_item *= 10;

	for (int i = 1; i <= current_objects.size(); i++) {
		for (int j = segments.size() + 1; j <= size; j++) {
			if (i == (j - segments.size())) {
				matrix[i][j] = E_t;
			}
			else {
				matrix[i][j] = -max_item;
			}
		}
	}

	for (int i = current_objects.size() + 1; i <= size; i++) {
		for (int j = 1; j <= segments.size(); j++) {
			if ((i - current_objects.size()) == j) {
				matrix[i][j] = E_s;
			}
			else {
				matrix[i][j] = -max_item;
			}
		}
	}

	for (int i = current_objects.size() + 1; i <= size; i++) {
		for (int j = segments.size() + 1; j <= size; j++) {
			matrix[i][j] = max_item;
		}
	}

	vector<int> objects_to_del;
	combination = HungarianAlgorithm<float>(matrix);
	
	if (is_first) {
		cout << ""; // << endl;
	}

	if (segments.size() != 0) {
		is_first = true;
		cout << endl << endl << "Non zero segments size: " << segments.size() << endl;
	}

	if (segments.size() == 2) {
		cout << "";
	}

	for (int i = 1; i < combination.size(); ++i)
	{
		size_t objID = combination[i] - 1, segID = i - 1;

		// if found object for segment
		// Shall we use !current_objects.empty() && ... ?
		//if (cosineSimilarity(current_objects[objID].feature, segments[segID].feature) >= similarityThreshold)
		//if (matrix[objID][segID] >= similarityThreshold)

		if (objID < current_objects.size() && (segID < segments.size()))
		{
			cout << objID << " <-> " << segID << " with ID: " << current_objects[objID].id << endl;
			current_objects[objID].pos = segments[segID].pos;
			current_objects[objID].feature = segments[segID].feature;
			Point center((segments[segID].pos[0].x + segments[segID].pos[1].x) / 2,
				(segments[segID].pos[0].y + segments[segID].pos[1].y) / 2);
			current_objects[objID].TrackerCounter = 0;
			current_objects[objID].trajectory.push_back(center);
		}

		// if not found any segment for �urrent object :
		else if (segID >= segments.size() && (objID < current_objects.size()))
		{
			cout << objID << " no segments for that" << " with ID: " << current_objects[objID].id << endl;
			current_objects[objID].TrackerCounter++;
			if (current_objects[objID].TrackerCounter >= 10)
			{
				objects_to_del.push_back(objID);
				break;
			}
		}
		else if (objID >= current_objects.size() && (segID < segments.size()))
		{
			Object new_obj = Object(segments[segID].pos, segments[segID].feature, next_id);
			cout << objID << "new one" << " with ID: " << next_id << endl;
			next_id++;
			current_objects.push_back(new_obj);
			//segments_centers[segID].obj = &current_objects.back();
		}
	}
	// if not found any segment for �urrent segment (new object on the picture):

	std::sort(objects_to_del.begin(), objects_to_del.end());
	std::reverse(objects_to_del.begin(), objects_to_del.end());
	for (auto& ind : objects_to_del)
	{
		current_objects.erase(current_objects.begin() + ind);
	}
	return current_objects;
}