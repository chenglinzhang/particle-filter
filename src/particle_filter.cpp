/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include <cassert>

#include "particle_filter.h"

#define NUM_PARTICLES 1000
#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized) return;

	// random number generator
	default_random_engine gen;

	// normal (Gaussian) distribution for x, y, and theta
	normal_distribution<double> dist_x(x, std[0]);	
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// number of particles to generate
	num_particles = NUM_PARTICLES;
	particles.resize(num_particles);

	// generate particles		
	for (auto &p: particles) {
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
	}

	// weights
	weights.resize(num_particles);

	// init flag up
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// random number generator
	default_random_engine gen;		

	// normal (Gaussian) distribution for x, y, and theta
	normal_distribution<double> dist_x(0.0, std_pos[0]);	
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (auto &p: particles) {
		// check yaw rate
		if (fabs(yaw_rate) < EPS) {
			// when the yaw rate is close to zero
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		} else {
			// when the yaw rate is not close to zero
			p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
		}			
		// add Gaussian noice
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// for each observation, find particle predicted in Nearest Neighbor distance
	for (auto &o: observations) {
	   	// init minimum distance to max double
		double min_distance = std::numeric_limits<double>::max();
		// find the closet prediction	   
	    for (auto &p: predicted) {
			double distance = dist(o.x, o.y, p.x, p.y);
			if (distance <= min_distance) {
				min_distance = distance;
				o.id = p.id;
			}
	    }
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];

		// 1. collect landmarks within sensor range
		std::vector<LandmarkObs> landmarks_in_range;
		for (auto &l: map_landmarks.landmark_list) {
			double distance = dist(p.x, p.y, l.x_f, l.y_f);
			if (distance < sensor_range) {
				landmarks_in_range.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
			}
		}

		// 2. transform observations to map coordinates
		std::vector<LandmarkObs> mapped_observations;
		for (auto &o: observations) {
			LandmarkObs t;
			t.x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
			t.y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
			mapped_observations.push_back(t);
		}

		// 3. associate observations with landmarks
		dataAssociation(landmarks_in_range, mapped_observations);

		// 4. update particle weight
		particles[i].weight = 1.0;	
		// p.weight = 1.0; // won't set particles[i].weight
		for (auto &m: mapped_observations) {
			// nearest landmark
			const LandmarkObs *nearest_landmark = nullptr;
			for (auto &l: landmarks_in_range) {
				if (m.id == l.id) {
					nearest_landmark = &l;
					break;
				}
			}
			// compute weight
			if (nearest_landmark != nullptr)
			{
				double xterm = pow(m.x - nearest_landmark->x, 2) / (2 * pow(std_landmark[0], 2));
				double yterm = pow(m.y - nearest_landmark->y, 2) / (2 * pow(std_landmark[1], 2));
				double w = exp(-(xterm + yterm)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
				// account for computing errors
				particles[i].weight *= (w == 0.0 ? EPS : w);	
				// p.weight *= (w == 0.0 ? EPS : w); // won't set particles[i].weight
			}
		}

		// keep updated weight
		weights[i] = particles[i].weight;
	}	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;
	std::mt19937 gen(rd());

	std::discrete_distribution<> dist(weights.begin(), weights.end());

	std::vector<Particle> resampled(num_particles);

	for (int i = 0; i < num_particles; i++) {
		int index = dist(gen);
		resampled[i] = particles[index];
	}

	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
