import React from 'react';
import nature from './assets/nature.png';
import heroBlue from './assets/hero-blue.png';
import skillsLogo from './assets/empower-skills.png';

export default function Hero() {
  return (
    <section className="hero">
      <div className="hero-row">
        <div className="hero-text">
          <h2>Grow Your Skills <br /> With Train Garden</h2>
          <img src={nature} alt="Nature" className="nature-image mobile-only" />
          <h3>
            Train Garden connects underskilled people <br />
            with free local and online opportunities like volunteering,
            community projects, and courses, to gain measurable skills, <br />
            and match them with eco-conscious companies.
          </h3>
        </div>

        <div className="hero-image-right desktop-only">
          <img src={nature} alt="Nature" className="nature-image" />
        </div>
      </div>

      <div className="hero-image-center">
        <img src={heroBlue} alt="Hero Blue" className="blue-image" />
        <img src={skillsLogo} alt="Skills Logo" className="skills-logo" />
      </div>
    </section>
  );
}