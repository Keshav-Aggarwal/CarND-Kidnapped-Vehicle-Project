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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if(is_initialized)
		return;
	default_random_engine gen;

	num_particles = 20;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	weights.resize(num_particles);

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for(int i = 0; i<num_particles;i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (int i=0; i < num_particles; i++)
	{
		double x, y, theta;

		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;

		if(fabs(yaw_rate) < 0.001)
		{
			particles[i].x += velocity * cos(theta) * (delta_t);
			particles[i].y += velocity * sin(theta) * (delta_t);
		}
		else
		{
			particles[i].x += (velocity / yaw_rate) * (sin(theta + (yaw_rate * delta_t)) - sin(theta));
			particles[i].y += (velocity / yaw_rate) * (cos(theta) - cos(theta + (yaw_rate * delta_t))) ;
			particles[i].theta += (yaw_rate * delta_t);
		}
		normal_distribution<double> std_x(0, std_pos[0]);
		normal_distribution<double> std_y(0, std_pos[1]);
		normal_distribution<double> std_theta(0, std_pos[2]);

		particles[i].x += std_x(gen);
		particles[i].y += std_y(gen);
		particles[i].theta += std_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int id;
	for(int i =0 ; i< observations.size(); i++)
	{
		LandmarkObs observation = observations[i];
		double min = numeric_limits<double>::max();

		for(int j =0; j < predicted.size(); j++)
		{
			LandmarkObs prediction = predicted[j];
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
			if(distance < min)
			{
				min = distance;
				id = j;
			}
		}
		observations[i].id = id;
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

	for(int i = 0; i < num_particles; i++){
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		//get the Prediction...
		std::vector<LandmarkObs> predictions;
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
			double distance= dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particles[i].x,  particles[i].y);

			if(distance < sensor_range)
			{
				LandmarkObs objPrediction;
				objPrediction.x = map_landmarks.landmark_list[j].x_f;
				objPrediction.y = map_landmarks.landmark_list[j].y_f;
				objPrediction.id = map_landmarks.landmark_list[j].id_i;

				predictions.push_back(objPrediction);
			}
		}

		//get the Observation...
		std::vector<LandmarkObs> transforms;
		for(int j = 0; j < observations.size(); j++){
			double xm = x + (cos(theta) * observations[j].x) - (sin(theta) * observations[j].y);
			double ym = y + (sin(theta) * observations[j].x) + (cos(theta) * observations[j].y);
			transforms.push_back(LandmarkObs({observations[j].id, xm, ym}));
		}

		dataAssociation(predictions, transforms);

		double prob = 1.0;
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		double norm = (1/(2 * M_PI * sig_x * sig_y));

		for(int j = 0; j < transforms.size(); j++){
			double o_x, o_y, l_x, l_y;
			o_x = transforms[j].x;
			o_y = transforms[j].y;

			l_x = predictions[transforms[j].id].x;
			l_y = predictions[transforms[j].id].y;

			double exp_w = exp(-1.0 * (((pow(o_x - l_x,2))/(2 * pow(sig_x,2))) + ((pow(o_y - l_y,2)/(2 * pow(sig_y,2))))));
			prob = prob * exp_w * norm;
		}
		particles[i].weight = prob;
		weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled;

	//Random device generator using std::discrete_distribution
	std::random_device rd;
	std::mt19937 gen(rd());

	discrete_distribution<int> random_number(0 , num_particles-1);
	discrete_distribution<int> disct_dis(weights.begin(), weights.end());

	unsigned index = random_number(rd);

	double max_weight = *max_element(weights.begin(), weights.end());

	//resampling
	double beta =0.0;
	for (int i =0; i <num_particles; i++)
	{
		beta+=disct_dis(gen) * 2.0 * max_weight;
		while(beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled.push_back(particles[index]);
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
	return particle;
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
