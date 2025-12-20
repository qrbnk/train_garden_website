import React from 'react';

// 1. Correctly import each image file as a module (using the relative path from Services.js)
import skillsEvaluation from './assets/skills-evaluation.png';
import trackProgress from './assets/track-progress.png';
import skillsHub from './assets/skills-hub.png';
import workshops from './assets/workshops.png';
import localOpportunities from './assets/local-opportunities.png';
import localRecruitment from './assets/local-recruitment.png';

export default function Services() {
  // 2. Use the imported module variables in the array
  const images = [
    { src: skillsEvaluation, alt: 'Skills Evaluation' },
    { src: trackProgress, alt: 'Track Progress' },
    { src: skillsHub, alt: 'Skills Hub' },
    { src: workshops, alt: 'Workshops' },
    { src: localOpportunities, alt: 'Local Opportunities' },
    { src: localRecruitment, alt: 'Local Recruitment' },
  ];

  return (
    <section className="services">
      {/* 3. The heading looks correct and fixes the previous warning */}
      <h2 className="section-title"> Our Services</h2>
      <div className="services-grid">
        {images.map((img, i) => (
          <div className="service-item" key={i}>
            {/* 4. The image usage is now correct: img.src holds the path created by Webpack */}
            <img src={img.src} alt={img.alt} className="services-image large" />
          </div>
        ))}
      </div>

      <div className="newsletter-container">
        <a href="/subscribe" className="newsletter-btn">Subscribe To Our Newsletter</a>
        <p>
          Join us on our mission to connect skills with opportunities. <br />
          Be part of the future!
        </p>
      </div>
    </section>
  );
}