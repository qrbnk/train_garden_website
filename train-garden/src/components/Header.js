import React from 'react';
import logo from './assets/logo.png'; // put your images in src/assets

export default function Header() {
  return (
    <header className="site-header container">
      <a href="/">
        <img src={logo} alt="Train Garden Logo" className="logo-image" />
      </a>
      <nav>
        <ul className="nav-links">
          
          {/* Line 15 */}
          <li><a href="#home">Home</a></li>
          
          {/* Line 18: Map Link (now simple text link) */}
          <li>
              <a 
                href="https://www.google.com/maps/d/u/1/embed?mid=1r0HJw53KoNuucSnhGcNFn5v5IHZ_U6w&ehbc=2E312F" width="640" height="480"
                target="_blank" 
                rel="noopener noreferrer" 
              >
                  Skills Hubs Around You
              </a>
          </li>
          
          {/* Line 27: Jobseekers CTA (still a button) */}
          <li><a href="https://forms.gle/RBbVRKBdVMtkdTSe9" target="_blank" rel="noopener noreferrer" className="cta">For Jobseekers</a></li>
          
          {/* Line 28: Contact Us Link (now simple text link) */}
          <li><a href="mailto:traingarden25@gmail.com?subject=Inquiry&body=Hello,%20I%20want%20to%20contact%20you" className="contact-btn">Contact Us</a></li>
          
        </ul>
      </nav>
    </header>
  );
}