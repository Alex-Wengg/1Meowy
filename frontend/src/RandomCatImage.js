import React, { useState, useEffect } from 'react';

function RandomCatImage() {
    const [imageSrc, setImageSrc] = useState('');

    useEffect(() => {
        const getRandomNumber = () => Math.floor(Math.random() * 22) + 1;

        const imagePath = `/cats/${getRandomNumber()}.png`;

        setImageSrc(imagePath);
    }, []);

    return (
            <img src={imageSrc} alt="Random Cat" style={{ width: '100%', height: 'auto', display: 'block' }} />
    );
}

export default RandomCatImage;
